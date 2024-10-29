"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import os

import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

from data import compile_data
from tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map)
from models import compile_model


# lidar_check 函数是一个可视化和分析 nuScenes 数据集中激光雷达和相机数据的工具。
# 它结合了图像和激光雷达点云的几何关系，并将结果绘制为鸟瞰视角（BEV）图像。
def lidar_check(version,   # version：数据集版本（如 'v1.0-trainval' 或 'v1.0-mini'）。
                dataroot='/data/nuscenes',  # dataroot：nuScenes 数据集的根目录。
                show_lidar=True,  # show_lidar：是否在图像上显示激光雷达点云。
                viz_train=False,  # viz_train：是否使用训练数据集进行可视化，否则使用验证数据集。
                nepochs=1,  # nepochs：循环遍历数据的次数。

                # 图像和数据增强参数（H, W, resize_lim,
                # final_dim, bot_pct_lim, rot_lim, rand_flip）：用于图像的尺寸调整和数据增强。
                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                # 栅格网格参数（xbound, ybound, zbound, dbound）：用于点云数据的边界设置
                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,    # bsz：批量大小。
                nworkers=10,    # nworkers：用于数据加载的并行进程数。
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='vizdata')

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val/3*2*rat*3, val/3*2*rat))
    gs = mpl.gridspec.GridSpec(2, 6, width_ratios=(1, 1, 1, 2*rat, 2*rat, 2*rat))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for epoch in range(nepochs):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs) in enumerate(loader):

            img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

            for si in range(imgs.shape[0]):
                plt.clf()
                final_ax = plt.subplot(gs[:, 5:6])
                for imgi, img in enumerate(imgs[si]):
                    ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(ego_pts, H, W)
                    plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    plt.imshow(showimg)
                    if show_lidar:
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask],
                                s=5, alpha=0.1, cmap='jet')
                    # plot_pts = post_rots[si, imgi].matmul(img_pts[si, imgi].view(-1, 3).t()) + post_trans[si, imgi].unsqueeze(1)
                    # plt.scatter(img_pts[:, :, :, 0].view(-1), img_pts[:, :, :, 1].view(-1), s=1)
                    plt.axis('off')

                    plt.sca(final_ax)
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.', label=cams[imgi].replace('_', ' '))
                
                plt.legend(loc='upper right')
                final_ax.set_aspect('equal')
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))

                ax = plt.subplot(gs[:, 3:4])
                plt.scatter(pts[si, 0], pts[si, 1], c=pts[si, 2], vmin=-5, vmax=5, s=5)
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))
                ax.set_aspect('equal')

                ax = plt.subplot(gs[:, 4:5])
                plt.imshow(binimgs[si].squeeze(0).T, origin='lower', cmap='Greys', vmin=0, vmax=1)

                imname = f'lcheck{epoch:03}_{batchi:05}_{si:02}.jpg'
                print('saving', imname)
                plt.savefig(imname)


# cumsum_check 函数用于测试和验证模型在使用累加计算时的表现。它的主要目的是比较模型在启用和禁用快速累加 (quickcumsum) 功能
# 时的前向传播和反向传播结果。这种对比可以帮助了解使用不同累加策略对模型输出和梯度的影响。
def cumsum_check(version,
                dataroot='/data/nuscenes',
                gpuid=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 6,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):

        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('autograd:    ', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('quick cumsum:', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())
        print()


# eval_model_iou 函数用于评估给定模型在验证数据集上的表现，主要计算模型的 IoU（Intersection over Union）指标，
# 用于评估 BEV（鸟瞰视角）语义分割任务的性能。IoU 是衡量语义分割质量的常用指标，评估模型在预测分割时的准确性。
def eval_model_iou(version,
                modelf,
                dataroot='/data/nuscenes',
                gpuid=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


# viz_model_preds 函数用于可视化模型在 nuScenes 数据集上的预测结果，特别是
# BEV（鸟瞰视角）语义分割的输出。它生成图像，展示模型的预测、输入图像以及静态地图信息，并将结果保存为图像文件。
def viz_model_preds(version,
                    modelf,  # modelf：模型文件路径，用于加载模型权重。
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=1, # gpuid：用于指定 GPU，如果小于 0 则使用 CPU。
                    viz_train=False,

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=4,
                    nworkers=10,
                    ):
    # grid_conf 和 data_aug_conf：定义了栅格配置和数据增强配置，用于数据加载和预处理。
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 6,
                }
    # 使用 compile_data 加载训练或验证数据集（根据 viz_train 参数选择）
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    # 调用 get_nusc_maps 加载 nuScenes 静态地图数据。
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    # device = torch.device('cuda')

    # 调用 compile_model 函数，构建一个基于给定配置的模型实例，
    # outC=1 表示模型的输出通道数为 1（通常用于单通道输出，例如 BEV 语义分割图）
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))   # load_state_dict() 用于将预训练权重加载到模型中。
    model.to(device)  # 将模型移动到指定的计算设备上（如 CPU 或 GPU）。
    print('loading model on:' + str(device))


    # 根据配置计算 BEV 栅格的参数，包括：
    # dx：栅格的尺寸（步长），用于确定 BEV 网格中每个单元格的大小。
    # bx：栅格的起始点，确定 BEV 网格的起点坐标。
    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()   # 将 dx 和 bx 转换为 numpy 数组。

    scene2map = {}  # scene2map 字典存储场景名和对应的地图位置。
    # 遍历数据集中的场景，为每个场景创建一个从场景名到地图名称的映射。
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']


    # 设置绘制图像的图形尺寸和布局参数
    val = 0.01
    fH, fW = final_dim
    # figsize：设置绘制图像的宽高比例。
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    # GridSpec：定义了绘制布局的网格规格，height_ratios 用于设置每个子图的高度比例
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    # update()：设置子图之间的间距和位置
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    # 创建输出文件夹
    output_dir = os.path.join(dataroot, 'bev_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 生成模型预测和绘制结果
    model.eval()
    counter = 0
    with torch.no_grad():   # 进入 no_grad() 上下文管理器，关闭梯度计算，以减少内存占用
        # 遍历数据加载器，加载一个批次的图像和对应的相机姿态数据，包括旋转、平移、内参、后处理旋转和平移矩阵、二进制图像等。
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            # 将图像及相机参数移动到指定设备上，并输入到模型中。
            # 模型输出的结果通过 sigmoid 函数进行归一化处理，并转换为 CPU 张量。
            out = out.sigmoid().cpu()

            # 遍历当前批次中的每个样本，并清空当前绘图，以便在新样本上重新绘制。
            for si in range(imgs.shape[0]):
                plt.clf()
                # 遍历当前样本中的每个摄像头图像，将图像绘制到子图中。
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)   # 使用 denormalize_img(img) 函数将图像反归一化，以便显示
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        # 如果摄像头图像在后 3 个位置（即后摄像头），则将图像进行左右翻转
                    plt.imshow(showimg)  # 显示反归一化后的图像，并关闭坐标轴。
                    plt.axis('off')    # 为图像添加注释，标记摄像头名称。
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                # 绘制 BEV 图像
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                # 设置一个新的子图用于绘制 BEV 图像，并关闭坐标轴。
                plt.setp(ax.spines.values(), color='b', linewidth=2)    # 设置子图边框颜色和宽度。
                plt.legend(handles=[    # 在子图中添加图例，用于标识车辆分割、车辆和地图信息。
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)   绘制地图
                rec = loader.dataset.ixes[counter]   # 获取当前样本的记录
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)   # 调用 plot_nusc_map 函数绘制 nuScenes 的地图。
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                # 将图像保存到指定文件夹
                imname = os.path.join(output_dir, f'eval{batchi:06}_{si:03}.jpg')
                print('saving', imname)
                plt.savefig(imname)
                counter += 1
