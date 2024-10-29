"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


# 用于加载和处理 nuScenes 数据集的 PyTorch 数据集类 NuscData
class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc   # nusc：nuScenes 数据对象。
        self.is_train = is_train   # is_train：指定当前数据集是用于训练还是验证。
        self.data_aug_conf = data_aug_conf    # data_aug_conf：数据增强配置，用于图像的预处理
        self.grid_conf = grid_conf    # grid_conf：栅格网格的配置，用于点云数据的处理

        self.scenes = self.get_scenes()  # get_scenes()：获取当前训练或验证的场景
        self.ixes = self.prepro()   # prepro()：根据场景过滤样本并按时间戳排序。

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']) # 计算网格的大小（dx）、偏移（bx）和网格数量（nx）。
        # gen_dx_bx() 函数用于根据 xbound、ybound 和 zbound 三个边界定义来生成栅格的基本属性，
        # 包括栅格的大小（dx）、偏移（bx）和数量（nx）。这些属性用于将原始的三维空间映射到栅格化的网格结构。
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()  # 修正 nuScenes 数据的文件路径，以便正确加载图像或点云。

        print(self)    # print(self) 是在控制台上打印当前类实例（NuscData 对象）的信息。

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.修正 nuScenes 数据的文件路径，以便正确加载图像或点云。
        用于调整 nuScenes 数据集中的文件路径，以确保图像或点云文件能够被正确加载。
        这是因为 nuScenes 数据集可能以不同的目录结构存储，因此在使用时需要对路径进行适当的修正。
        """
        # check if default file paths work
        rec = self.ixes[0]  # rec = self.ixes[0]：获取数据集中第一个样本的索引，self.ixes 是按时间戳排序的样本索引列表。
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])  # 获取与该样本关联的前视摄像头图像数据。
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])  # 拼接出图像的完整路径，self.nusc.dataroot 是 nuScenes 数据的根目录。

        def find_name(f):
            # find_name(f)：辅助函数，用于从给定路径 f 中提取层次结构，并返回分解后的文件夹名和文件名。
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # get_scenes() 方法用于根据 nuScenes 数据集的版本和划分（训练或验证），选择并返回对应的场景列表。
        # filter by scene split    根据版本和数据集划分（训练或验证），选择相应的场景。
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]    # 根据 self.nusc.version 和 self.is_train 设置合适的场景划分

        scenes = create_splits_scenes()[split]
        # 调用 create_splits_scenes() 函数获取场景划分的字典，
        # 其中每个键对应不同划分（如 'train'、'val'、'mini_train'、'mini_val'），而值则是对应的场景列表。

        return scenes

    def prepro(self):   #  过滤出当前场景的样本，并按时间戳排序以保持时间序列的一致性。
        samples = [samp for samp in self.nusc.sample]  # 从 self.nusc.sample 中获取所有样本，将其存储在列表 samples 中
        # self.nusc.sample 是 nuScenes 数据集中所有样本的列表，其中每个样本包含场景、时间戳和其他相关信息

        # remove samples that aren't in this split
        # 遍历所有样本，并通过 samp['scene_token'] 获取样本所属场景的 token。
        # self.nusc.get('scene', samp['scene_token'])['name']：根据场景 token 获取场景的名称
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        # 按场景 token 和时间戳对样本进行排序，以确保样本按照时间顺序排列
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):    # 该方法根据训练或验证模式进行图像的随机裁剪、缩放、翻转和旋转。
        # H, W：原始图像的高度和宽度。
        # fH, fW：目标图像的最终尺寸（final_dim），用于裁剪后的图像输出。
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        # 训练模式下的数据增强
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            # resize：从 resize_lim 范围内随机选取一个缩放比例，用于图像缩放
            resize_dims = (int(W*resize), int(H*resize))
            # resize_dims：计算缩放后的图像尺寸
            newW, newH = resize_dims

            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            # crop_h：计算裁剪区域的垂直偏移量，基于 bot_pct_lim 的随机值
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            # crop_w：随机确定水平偏移量
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # crop：定义裁剪区域，格式为 (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                # flip：随机决定是否水平翻转图像
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
            # rotate：在 rot_lim 范围内随机选择旋转角度
        else:
            # 当 self.is_train 为 False 时，执行验证模式下的固定数据增强
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims

            # 中心裁剪
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        # 该方法加载图像数据并进行数据增强，同时返回与相机相关的内参、旋转矩阵、平移向量和后处理参数。

        # 初始化多个空列表，用于存储图像、旋转矩阵、平移向量、内参矩阵、增强后的旋转矩阵和平移向量。
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:   # 遍历给定的相机列表 cams，依次对每个相机的图像数据进行处理
            samp = self.nusc.get('sample_data', rec['data'][cam])   # samp：从样本记录中获取指定相机的图像数据
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])   # imgname：拼接出图像的完整路径
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 初始化 post_rot 为单位矩阵（2x2），post_tran 为零向量（2维），用于后续的数据增强变换

            # sens：获取相机的标定信息
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])  # 将相机的内参（camera_intrinsic）转换为张量
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)   # 将相机的旋转四元数转换为旋转矩阵张量
            tran = torch.Tensor(sens['translation'])   # tran：将相机的平移向量转换为张量

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # 调用 img_transform() 方法，对图像进行实际的增强操作，并得到增强后的图像（img）、旋转变换矩阵（post_rot2）和平移变换向量（post_tran2）
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3  将增强矩阵扩展为 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # print('\n')
            # # print(intrin)
            # # print(rot)
            # # print(tran)
            # # print(post_rot)
            # # print(post_tran)
            # print('\n')

            # 分别添加到相应的列表中
            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):    # 该方法加载指定数量的激光雷达扫描数据，并返回点云的三维坐标。
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):    # 根据标注信息创建一个二值图像，其中车辆被标记为 1。
        # get_binimg() 方法用于根据给定样本的标注信息创建一个二值图像（Binary Image），
        # 其中车辆区域被标记为 1，其他区域为 0。该二值图像主要用于在训练或推理过程中对感兴趣的车辆区域进行标注
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        # egopose：获取自车（ego vehicle）的位置信息与姿态信息
        # egopose['translation']：自车的平移向量。
        # egopose['rotation']：自车的旋转四元数
        trans = -np.array(egopose['translation'])    # trans：取负值的平移向量，用于将世界坐标系中的点转换到自车坐标系
        rot = Quaternion(egopose['rotation']).inverse    # rot：计算旋转四元数的逆，表示将标注框旋转到自车坐标系
        img = np.zeros((self.nx[0], self.nx[1]))  # 创建一个大小为 (nx[0], nx[1]) 的二值图像，初始值为 0
        # self.nx 是栅格网格的尺寸参数，nx[0] 是图像的高度，nx[1] 是图像的宽度
        for tok in rec['anns']:    # 遍历样本中的每个标注 token，并获取对应的实例信息（inst）
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                # if not inst['category_name'].split('.')[0] == 'vehicle'：只处理类别为 'vehicle' 的标注，其他类别将被跳过
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))  # Box()：创建一个标注框对象，基于标注的平移（translation）、大小（size）和旋转（rotation）信息
            box.translate(trans)   # 将标注框平移到自车坐标系
            box.rotate(rot)     # 将标注框旋转到自车坐标系

            pts = box.bottom_corners()[:2].T   # box.bottom_corners()[:2].T：获取标注框底部角点的 x 和 y 坐标，并进行转置
            pts = np.round(   # pts = np.round(...)：将角点坐标从世界坐标系映射到栅格坐标系
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):   # 随机选择使用的相机，以进行多视角数据的增强。
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):   # VizData 类是基于 NuscData 类的一个子类，它重写了 __getitem__() 方法，
    # 用于可视化和处理 nuScenes 数据集。该类继承了 NuscData 的所有属性和方法，并在其基础上实现了对数据的读取
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):  # __getitem__() 是 PyTorch 数据集类中的核心方法，用于根据索引 index 返回一个样本数据
        rec = self.ixes[index]   # rec = self.ixes[index]：从样本索引列表 self.ixes 中获取第 index 个样本记录
        
        cams = self.choose_cams()  # 调用 choose_cams() 方法选择相机（cams），用于确定读取哪些相机的图像数据
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)   # nsweeps=3 表示使用 3 帧激光雷达扫描数据进行融合，以提升感知效果
        binimg = self.get_binimg(rec)

        print('\n')
        print(imgs)
        print(rots)
        print(trans)
        print(intrins)
        print(post_rots)
        print(post_trans)
        print(lidar_data)
        print(binimg)
        print('\n')
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    #  SegmentationData 类是基于 NuscData 的子类，重写了 __getitem__() 方法，用于提取 nuScenes 数
    #  据集中的图像数据和 BEV（鸟瞰视角）分割图像。这一类适用于 BEV 语义分割任务，通过返回图像、相机参数
    #  以及分割标签图来实现自动驾驶环境中的场景理解。
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)

        print('\n')
        print(imgs)
        print(rots)
        print(trans)
        print(intrins)
        print(post_rots)
        print(post_trans)
        print(binimg)
        print('\n')
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    # worker_rnd_init 是一个用于初始化工作进程的函数，主要用于在数据加载时确保每个工作进程（worker）的随
    # 机数生成器有不同的种子，从而避免在数据增强或采样时的随机性一致性问题。
    np.random.seed(13 + x)



def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    # compile_data 函数用于创建用于训练和验证的 PyTorch 数据加载器（DataLoader），它将 nuScenes 数据集
    # 加载为指定格式，并根据数据增强和网格配置生成可用于训练的批数据。
    nusc = NuScenes(version='v1.0-{}'.format(version),   # NuScenes：加载 nuScenes 数据集对象
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]    # 根据 parser_name，从字典中选择适当的解析器类
    #  parser结果就是VizData函数或者SegmentationData函数
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
