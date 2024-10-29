
from explore import viz_model_preds

def main():
    # 参数设置
    version = 'mini'  # 使用 nuScenes-mini 数据集
    modelf = 'model525000.pt'  # 替换为实际模型权重路径
    dataroot = './nuscenes'  # nuScenes-mini 数据集的根目录
    map_folder = './nuscenes/mini'  # nuScenes-mini 静态地图数据路径
    gpuid = 0  # 设置为0表示使用第一个GPU, -1表示使用CPU
    viz_train = False  # 设置为False表示使用验证集进行可视化

    # 图像和数据增强配置
    H = 900
    W = 1600
    resize_lim = (0.193, 0.225)
    final_dim = (128, 352)
    bot_pct_lim = (0.0, 0.22)
    rot_lim = (-5.4, 5.4)
    rand_flip = True

    # 栅格网格参数
    xbound = [-50.0, 50.0, 0.5]
    ybound = [-50.0, 50.0, 0.5]
    zbound = [-10.0, 10.0, 20.0]
    dbound = [4.0, 45.0, 1.0]

    bsz = 4  # 批量大小
    nworkers = 10  # 数据加载的工作进程数

    # 调用可视化函数
    viz_model_preds(
        version=version,
        modelf=modelf,
        dataroot=dataroot,
        map_folder=map_folder,
        gpuid=gpuid,
        viz_train=viz_train,
        H=H, W=W,
        resize_lim=resize_lim,
        final_dim=final_dim,
        bot_pct_lim=bot_pct_lim,
        rot_lim=rot_lim,
        rand_flip=rand_flip,
        xbound=xbound,
        ybound=ybound,
        zbound=zbound,
        dbound=dbound,
        bsz=bsz,
        nworkers=nworkers
    )

if __name__ == "__main__":
    main()
