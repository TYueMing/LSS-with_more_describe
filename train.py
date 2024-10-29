"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


def train(version,    # version：指定 nuScenes 数据集的版本（如 'mini' 或 'v1.0-trainval'）
            dataroot='/data/nuscenes',    # dataroot：nuScenes 数据集的根目录
            nepochs=10000,    # 训练的总轮数
            gpuid=1,   # gpuid：指定使用的 GPU（若为负数，则使用 CPU）。

            H=900, W=1600,    # H、W：输入图像的高度和宽度
            resize_lim=(0.193, 0.225),    # resize_lim：数据增强时的随机缩放范围
            final_dim=(128, 352),     # final_dim：输入图像最终的分辨率
            bot_pct_lim=(0.0, 0.22),     # bot_pct_lim：数据增强时裁剪底部的比例范围
            rot_lim=(-5.4, 5.4),     # rot_lim：随机旋转的范围
            rand_flip=True,     #  rand_flip：是否随机水平翻转图像
            ncams=5, ####################################################
          # 这里的想法是，通过使用内在函数、外函数、预测的分类深度和总和池化，
          # 理论上可以将来自任意数量的相机 （N） 的特征投影到以机器人为中心的 3D 空间。
          # NuScenes 的情况限制我们在 1 到 6 之间进行选择，但该方法理论上适用于任意数量的摄像机
          # （例如 1000 台摄像机）。在实践中，此相机数量将受到您的计算能力 （GPU 内存） 和您希望系统运行速度的限制。
            max_grad_norm=5.0,    # max_grad_norm：最大梯度范数（用于梯度裁剪）
            pos_weight=2.13,    # pos_weight：用于加权交叉熵损失的正类权重
            logdir='./runs',     # logdir：TensorBoard 日志的保存路径

            # BEV 网格的边界范围
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,    # 训练的批量大小
            nworkers=10,     # 数据加载时的工作进程数
            lr=1e-3,      # 学习率
            weight_decay=1e-7,      # 权重衰减系数
            ):
    ############ 生成网格配置
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    # ############### 数据增强配置
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }

    ##################### 加载数据
    # 调用 compile_data 函数加载训练和验证数据集：
    #     根据 data_aug_conf 和 grid_conf 生成训练数据和验证数据。
    #     使用 bsz 和 nworkers 控制批量大小和数据加载的进程数。
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    ############### 调用 compile_model 函数来创建模型实例，outC=1 表示输出通道数为 1
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    ## 使用 Adam 优化器来优化模型参数，学习率为 lr，权重衰减系数为 weight_decay
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ## 使用自定义的交叉熵损失 SimpleLoss，并将损失函数加载到 GPU 上
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    ## 创建 TensorBoard 日志记录器，用于记录训练过程中损失、学习率等信息
    writer = SummaryWriter(logdir=logdir)
    # 设置验证的步长（即每隔多少步进行一次验证）：
    #     如果是 mini 数据集，则设置为 1000 步；
    #     否则设置为 10000 步。
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
