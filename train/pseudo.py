import _init_paths

import argparse
from datetime import date, datetime
import logging
import os
import shutil

# from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import SuperCLEVRTrain, ToTensor, Normalize, InfiniteSampler
from src.models import NearestMemoryManager, NetE2E, mask_remove_near
from src.utils import str2bool, load_off
from src.datasets.pascal3dp_train import PASCAL3DPTrain
from src.datasets.oodcv_train import OODCVTrain


def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised domain adaptation')

    # General args
    parser.add_argument('--exp_name', type=str, default='P3D-Diffusion_car')
    parser.add_argument('--exp_type', type=str, default='pseudo', choices=['pseudo', 'fine_tune'])
    parser.add_argument('--num', type=int, default=92)
    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='../experiments')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    # Model args
    parser.add_argument('--backbone', type=str, default='resnetext')
    parser.add_argument('--d_feature', type=int, default=128)
    parser.add_argument('--local_size', type=int, default=1)
    parser.add_argument('--separate_bank', type=str2bool, default=False)
    parser.add_argument('--max_group', type=int, default=512)

    # Data args
    parser.add_argument('--mesh_path', type=str, default='../data/CAD_cate_pascal')
    parser.add_argument('--dataset_path', type=str, default='../data/car')
    parser.add_argument('--distance', type=float, default=6)

    parser.add_argument('--crop_object', type=str2bool, default=False)
    parser.add_argument('--rotate', type=float, default=2)
    parser.add_argument('--bg_path', type=str, default='../data/bg')
    parser.add_argument('--filename_prefix', type=str, default='P3D-Diffusion')
    parser.add_argument('--workers', type=str, default=4)

    # Training args
    parser.add_argument('--prev_ckpt', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--log_itr', type=int, default=100)
    parser.add_argument('--save_itr', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--update_lr_itr', type=int, default=30000)
    parser.add_argument('--update_lr_ratio', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.92)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--train_accumulate', type=int, default=10)
    parser.add_argument('--distance_thr', type=int, default=48)
    parser.add_argument('--weight_noise', type=float, default=0.005)
    parser.add_argument('--num_noise', type=int, default=5)
    parser.add_argument('--T', type=float, default=0.07)
    parser.add_argument('--adj_momentum', type=float, default=0.96)

    args = parser.parse_args()

    return args


def reserve_gpu():
    a = torch.zeros((1500, 1000, 2000), dtype=torch.float64, device='cuda:0')
    del a


def prepare_data_pseudo(args):
    train_transform = transforms.Compose([ToTensor(), Normalize()])

    npy = np.load(f'../test_pascal3d/exp/{args.category}.npy', allow_pickle=True)[()]

    train_dataset = PASCAL3DPTrain(
        img_path=f'../PASCAL3D+_release1.1/Images/{args.category}_imagenet',
        anno_path=f'../PASCAL3D+_release1.1/Annotations/{args.category}_imagenet',
        list_file=f'../test_pascal3d/{args.category}_100.txt',
        category=f'{args.category}',
        crop_object=True,
        bg_path='../data/bg',
        mesh_path='../data/CAD_cate_pascal',
        transform=train_transform,
        dist=args.distance,
        pseudo_label=npy['pred'],
    )

    train_sampler = InfiniteSampler(dataset=train_dataset, shuffle=True, seed=args.seed, window_size=0.5)
    train_iterator = iter(
        DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers))

    if args.val:
        val_transform = transforms.Compose([ToTensor(), Normalize()])
        val_dataset = SuperCLEVRTrain(
            img_path=args.val_img_path,
            anno_path=args.val_anno_path,
            prefix=f'{args.filename_prefix}_val',
            category=args.category,
            transform=val_transform,
            enable_cache=False
        )
        val_iterator = iter(
            DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers))
    else:
        val_dataset, val_iterator = None, None

    return train_dataset, train_iterator, val_dataset, val_iterator


def prepare_data_real(args):
    train_transform = transforms.Compose([ToTensor(), Normalize()])

    train_dataset = PASCAL3DPTrain(
        img_path=f'../PASCAL3D+_release1.1/Images/{args.category}_imagenet',
        anno_path=f'../PASCAL3D+_release1.1/Annotations/{args.category}_imagenet',
        list_file=f'../PASCAL3D+_release1.1/Image_sets/{args.category}_imagenet_train.txt',
        category=f'{args.category}',
        crop_object=True,
        bg_path='../data/bg',
        mesh_path='../data/CAD_cate_pascal',
        transform=train_transform,
        dist=args.distance,
    )

    np.random.seed(1)
    np.random.shuffle(train_dataset.file_list)
    # args.num_samples = 500
    # if args.num_samples > 0 and args.num_samples < len(train_dataset.file_list):
    # num = len(train_dataset)
    train_dataset.file_list = train_dataset.file_list[:args.num]
    train_sampler = InfiniteSampler(dataset=train_dataset, shuffle=True, seed=args.seed, window_size=0.5)
    train_iterator = iter(
        DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers))

    if args.val:
        val_transform = transforms.Compose([ToTensor(), Normalize()])
        val_dataset = SuperCLEVRTrain(
            img_path=args.val_img_path,
            anno_path=args.val_anno_path,
            prefix=f'{args.filename_prefix}_val',
            category=args.category,
            transform=val_transform,
            enable_cache=False
        )
        val_iterator = iter(
            DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers))
    else:
        val_dataset, val_iterator = None, None

    return train_dataset, train_iterator, val_dataset, val_iterator


def prepare_data_ood(args):
    train_transform = transforms.Compose([ToTensor(), Normalize()])

    train_dataset = OODCVTrain(
        img_path=f'/home/jiahao/pretrain_6d_pose/OOD_train/images/{args.category}',
        anno_path=f'/home/jiahao/pretrain_6d_pose/OOD_train/annotations/{args.category}',
        # file_list=f'/home/jiahao/pretrain_6d_pose/test_pascal3d/{args.category}_pseudo_.txt',
        category=f'{args.category}',
        crop_object=True,
        bg_path='/home/jiahao/pretrain_6d_pose/data/bg',
        mesh_path='/home/jiahao/pretrain_6d_pose/data/CAD_cate_pascal',
        transform=train_transform,
        dist=args.distance,
        # pseudo=npy
    )

    # print(len(train_dataset))
    np.random.seed(1)
    np.random.shuffle(train_dataset.file_list)
    # args.num_samples = 500
    # if args.num_samples > 0 and args.num_samples < len(train_dataset.file_list):
    num = len(train_dataset)
    train_dataset.file_list = train_dataset.file_list[:82]
    train_sampler = InfiniteSampler(dataset=train_dataset, shuffle=True, seed=args.seed, window_size=0.5)
    train_iterator = iter(
        DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers))

    if args.val:
        val_transform = transforms.Compose([ToTensor(), Normalize()])
        val_dataset = SuperCLEVRTrain(
            img_path=args.val_img_path,
            anno_path=args.val_anno_path,
            prefix=f'{args.filename_prefix}_val',
            category=args.category,
            transform=val_transform,
            enable_cache=False
        )
        val_iterator = iter(
            DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers))
    else:
        val_dataset, val_iterator = None, None

    return train_dataset, train_iterator, val_dataset, val_iterator


def train_one_step(net, memory_bank, train_iterator, criterion, optimizer, args, itr):
    sample = next(train_iterator)
    img, kp, kpvis, obj_mask, distance = sample['img'], sample['kp'], sample['kpvis'], sample['obj_mask'], sample[
        'distance']
    img, kp, kpvis, obj_mask, distance = img.cuda(), kp.cuda(), kpvis.cuda(), obj_mask.cuda(), distance.cuda()
    # num_objs = sample['num_objs']
    num_objs = 1
    kp = kp[:, :, [1, 0]]

    y_num = args.n
    index = torch.Tensor([[k for k in range(y_num)]] * img.shape[0])
    index = index.cuda()

    features = net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask.float())

    get, y_idx, noise_sim = memory_bank(features, index, kpvis)
    get /= args.T
    mask_distance_legal = mask_remove_near(kp, thr=args.distance_thr * 5.0 / distance,
                                           num_neg=args.num_noise * args.max_group,
                                           dtype_template=get, neg_weight=args.weight_noise)

    kpvis = kpvis.type(torch.bool).to(kpvis.device)
    loss = criterion(((get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2])))[kpvis.view(-1), :],
                     y_idx.view(-1)[kpvis.view(-1)])
    loss = torch.mean(loss)

    loss_main = loss.item()
    if args.num_noise > 0 and True:
        loss_reg = torch.mean(noise_sim) * 0.1
        loss += loss_reg
    else:
        loss_reg = torch.zeros(1)
    loss.backward()

    if itr % args.train_accumulate == 0:
        optimizer.step()
        optimizer.zero_grad()

    return {'loss': loss.item(), 'loss_main': loss_main, 'loss_reg': loss_reg.item(),
            'lr': optimizer.param_groups[0]['lr']}


def val_one_step(net, memory_bank, val_dataloader, criterion, args, itr):
    # if not args.val:
    #     return {'loss': 0.0, 'loss_main': 0.0, 'loss_reg': 0.0}
    # else:
    #     raise NotImplementedError
    return {'loss': 0.0, 'loss_main': 0.0, 'loss_reg': 0.0}


def pseudo(args):
    cate = args.category
    args.prev_ckpt = f'../experiments/P3D-Diffusion_{cate}/ckpts/saved_model_2000.pth'
    if cate in ['bus', 'car', 'motorbike', 'aeroplane', 'bicycle', 'boat', 'diningtable', 'train']:
        args.distance = 6
    if cate in ['chair', 'bottle']:
        args.distance = 10
    if cate in ['sofa', 'tvmonitor']:
        args.distance = 8

    args.mesh_path = os.path.join(args.mesh_path, args.category, '01.off')
    args.train_img_path = os.path.join(args.dataset_path, 'train', 'images')
    args.train_anno_path = os.path.join(args.dataset_path, 'train', 'annotations')
    args.val_img_path = os.path.join(args.dataset_path, 'val', 'images')
    args.val_anno_path = os.path.join(args.dataset_path, 'val', 'annotations')
    args.val_img_path = None
    args.val_anno_path = None
    args.val = args.val_img_path is not None and args.val_anno_path is not None

    print(args)

    save_path = os.path.join(args.save_path, args.exp_name)
    ckpt_path = os.path.join(save_path, 'ckpts')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    # shutil.copyfile(__file__, os.path.join(save_path, __file__))
    load_prev_ckpt = False
    if args.prev_ckpt is not None:
        load_prev_ckpt = True
        prev_ckpt = args.prev_ckpt

    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(save_path, 'log.txt'),
                        filemode='a' if load_prev_ckpt else 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(args)

    net = NetE2E(net_type=args.backbone, local_size=[args.local_size, args.local_size],
                 output_dimension=args.d_feature,
                 reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=True)
    logging.info(f'num params {sum(p.numel() for p in net.net.parameters())}')
    net = nn.DataParallel(net).cuda().train()
    if load_prev_ckpt:
        checkpoint = torch.load(prev_ckpt, map_location=args.device)
        net.load_state_dict(checkpoint['state'])
        print('load_prev_ckpt done')

    args.n = load_off(args.mesh_path)[0].shape[0]

    memory_bank = NearestMemoryManager(inputSize=args.d_feature,
                                       outputSize=args.n + args.num_noise * args.max_group,
                                       K=1, num_noise=args.num_noise, num_pos=args.n, momentum=args.adj_momentum)
    # memory_bank = memory_bank.cuda()
    if load_prev_ckpt:
        with torch.no_grad():
            # memory_bank.memory.copy_(checkpoint['memory'][0:memory_bank.memory.shape[0]])
            memory_bank.memory.copy_(checkpoint['memory'])
            print('load_memory done')

    memory_bank = memory_bank.cuda()

    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset, train_iterator, val_dataset, val_iterator = prepare_data_pseudo(args)
    logging.info(f'found {len(train_dataset)} for training ({args.category})')
    if args.val:
        logging.info(f'found {len(val_dataset)} for validation ({args.category})')

    # comet = Experiment(api_key='UKaGtn0xxEiEIPoG70XmgJhAB', project_name='nemo6d', workspace='wufeim')
    comet = None

    if comet:
        comet.set_name(args.exp_name)
        comet.log_parameters(vars(args))
        logging.info(comet.url)

    logging.info('Start training:')
    logging.info(f'Experiment:     {args.exp_name}')
    logging.info(f'Category:       {args.category}')
    logging.info(f'Num imgs train: {len(train_dataset)}')
    if args.val:
        logging.info(f'Num imgs val:   {len(val_dataset)}')
    logging.info(f'Total itr:      {args.iterations}')
    logging.info(f'LR:             {args.lr}')
    logging.info(f'Update LR itr:  {args.update_lr_itr}')
    logging.info(f'Updated LR:     {args.lr * args.update_lr_ratio}')

    log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
    log_val_loss, log_val_loss_main, log_val_loss_reg = [], [], []
    lr = args.lr

    for itr in range(args.iterations):
        train_log_dict = train_one_step(net, memory_bank, train_iterator, criterion, optim, args, itr)
        val_log_dict = val_one_step(net, memory_bank, val_iterator, criterion, args, itr)

        log_train_loss.append(train_log_dict['loss'])
        log_train_loss_main.append(train_log_dict['loss_main'])
        log_train_loss_reg.append(train_log_dict['loss_reg'])

        log_val_loss.append(val_log_dict['loss'])
        log_val_loss_main.append(val_log_dict['loss_main'])
        log_val_loss_reg.append(val_log_dict['loss_reg'])

        if itr == 0 or (itr + 1) % args.log_itr == 0:
            train_loss, train_loss_main, train_loss_reg = np.mean(log_train_loss), np.mean(
                log_train_loss_main), np.mean(log_train_loss_reg)
            val_loss, val_loss_main, val_loss_reg = np.mean(log_val_loss), np.mean(log_val_loss_main), np.mean(
                log_val_loss_reg)
            if itr > 0:
                log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
                log_val_loss, log_val_loss_main, log_val_loss_reg = [], [], []
            if comet:
                comet.log_metric('lr', train_log_dict['lr'], step=itr + 1)
                comet.log_metric('train_loss', train_loss, step=itr + 1)
                comet.log_metric('train_loss_main', train_loss_main, step=itr + 1)
                comet.log_metric('train_loss_reg', train_loss_reg, step=itr + 1)
                comet.log_metric('val_loss', val_loss, step=itr + 1)
                comet.log_metric('val_loss_main', val_loss_main, step=itr + 1)
                comet.log_metric('val_loss_reg', val_loss_reg, step=itr + 1)

            logging.info(
                f'[Itr {itr + 1}] lr={train_log_dict["lr"]} loss={train_loss:.5f} loss_main={train_loss_main:.5f} loss_reg={train_loss_reg:.5f}')

        if (itr + 1) % args.save_itr == 0:
            ckpt = {}
            ckpt['state'] = net.state_dict()
            ckpt['memory'] = memory_bank.memory
            ckpt['timestamp'] = int(datetime.timestamp(datetime.now()))
            ckpt['args'] = vars(args)
            ckpt['step'] = itr + 1
            ckpt['lr'] = train_log_dict["lr"]
            torch.save(ckpt, os.path.join(save_path, 'ckpts', f'saved_model_{itr + 1}.pth'))

        if (itr + 1) % args.update_lr_itr == 0:
            lr = lr * args.update_lr_ratio
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            logging.info(f'update learning rate: {lr}')


def fine_tune(args):
    cate = args.category
    args.exp_name = f'{cate}'

    args.prev_ckpt = f'../experiments/P3D-Diffusion_{cate}/ckpts/saved_model_2000.pth'
    if cate in ['bus', 'car', 'motorbike', 'aeroplane', 'bicycle', 'boat', 'diningtable', 'train']:
        args.distance = 6
    if cate in ['chair', 'bottle']:
        args.distance = 10
    if cate in ['sofa', 'tvmonitor']:
        args.distance = 8

    args.mesh_path = os.path.join(args.mesh_path, args.category, '01.off')
    args.train_img_path = os.path.join(args.dataset_path, 'train', 'images')
    args.train_anno_path = os.path.join(args.dataset_path, 'train', 'annotations')
    args.val_img_path = os.path.join(args.dataset_path, 'val', 'images')
    args.val_anno_path = os.path.join(args.dataset_path, 'val', 'annotations')
    args.val_img_path = None
    args.val_anno_path = None
    args.val = args.val_img_path is not None and args.val_anno_path is not None

    print(args)

    save_path = os.path.join(args.save_path, args.exp_name)
    ckpt_path = os.path.join(save_path, 'ckpts')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    #shutil.copyfile(__file__, os.path.join(save_path, __file__))
    load_prev_ckpt = False
    if args.prev_ckpt is not None:
        load_prev_ckpt = True
        prev_ckpt = args.prev_ckpt

    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(save_path, 'log.txt'),
                        filemode='a' if load_prev_ckpt else 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(args)

    net = NetE2E(net_type=args.backbone, local_size=[args.local_size, args.local_size],
                 output_dimension=args.d_feature,
                 reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=True)
    logging.info(f'num params {sum(p.numel() for p in net.net.parameters())}')
    net = nn.DataParallel(net).cuda().train()
    if load_prev_ckpt:
        checkpoint = torch.load(prev_ckpt, map_location=args.device)
        net.load_state_dict(checkpoint['state'])
        print('load_prev_ckpt done')

    args.n = load_off(args.mesh_path)[0].shape[0]

    memory_bank = NearestMemoryManager(inputSize=args.d_feature,
                                       outputSize=args.n + args.num_noise * args.max_group,
                                       K=1, num_noise=args.num_noise, num_pos=args.n,
                                       momentum=args.adj_momentum)
    # memory_bank = memory_bank.cuda()
    if load_prev_ckpt:
        with torch.no_grad():
            # memory_bank.memory.copy_(checkpoint['memory'][0:memory_bank.memory.shape[0]])
            memory_bank.memory.copy_(checkpoint['memory'])
            print('load_memory done')

    memory_bank = memory_bank.cuda()

    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset, train_iterator, val_dataset, val_iterator = prepare_data_real(args)
    logging.info(f'found {len(train_dataset)} for training ({args.category})')
    if args.val:
        logging.info(f'found {len(val_dataset)} for validation ({args.category})')

    # comet = Experiment(api_key='UKaGtn0xxEiEIPoG70XmgJhAB', project_name='nemo6d', workspace='wufeim')
    comet = None

    if comet:
        comet.set_name(args.exp_name)
        comet.log_parameters(vars(args))
        logging.info(comet.url)

    logging.info('Start training:')
    logging.info(f'Experiment:     {args.exp_name}')
    logging.info(f'Category:       {args.category}')
    logging.info(f'Num imgs train: {len(train_dataset)}')
    if args.val:
        logging.info(f'Num imgs val:   {len(val_dataset)}')
    logging.info(f'Total itr:      {args.iterations}')
    logging.info(f'LR:             {args.lr}')
    logging.info(f'Update LR itr:  {args.update_lr_itr}')
    logging.info(f'Updated LR:     {args.lr * args.update_lr_ratio}')

    log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
    log_val_loss, log_val_loss_main, log_val_loss_reg = [], [], []
    lr = args.lr

    for itr in range(args.iterations):
        train_log_dict = train_one_step(net, memory_bank, train_iterator, criterion, optim, args, itr)
        val_log_dict = val_one_step(net, memory_bank, val_iterator, criterion, args, itr)

        log_train_loss.append(train_log_dict['loss'])
        log_train_loss_main.append(train_log_dict['loss_main'])
        log_train_loss_reg.append(train_log_dict['loss_reg'])

        log_val_loss.append(val_log_dict['loss'])
        log_val_loss_main.append(val_log_dict['loss_main'])
        log_val_loss_reg.append(val_log_dict['loss_reg'])

        if itr == 0 or (itr + 1) % args.log_itr == 0:
            train_loss, train_loss_main, train_loss_reg = np.mean(log_train_loss), np.mean(
                log_train_loss_main), np.mean(log_train_loss_reg)
            val_loss, val_loss_main, val_loss_reg = np.mean(log_val_loss), np.mean(log_val_loss_main), np.mean(
                log_val_loss_reg)
            if itr > 0:
                log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
                log_val_loss, log_val_loss_main, log_val_loss_reg = [], [], []
            if comet:
                comet.log_metric('lr', train_log_dict['lr'], step=itr + 1)
                comet.log_metric('train_loss', train_loss, step=itr + 1)
                comet.log_metric('train_loss_main', train_loss_main, step=itr + 1)
                comet.log_metric('train_loss_reg', train_loss_reg, step=itr + 1)
                comet.log_metric('val_loss', val_loss, step=itr + 1)
                comet.log_metric('val_loss_main', val_loss_main, step=itr + 1)
                comet.log_metric('val_loss_reg', val_loss_reg, step=itr + 1)

            logging.info(
                f'[Itr {itr + 1}] lr={train_log_dict["lr"]} loss={train_loss:.5f} loss_main={train_loss_main:.5f} loss_reg={train_loss_reg:.5f}')

        if (itr + 1) % args.save_itr == 0:
            ckpt = {}
            ckpt['state'] = net.state_dict()
            ckpt['memory'] = memory_bank.memory
            ckpt['timestamp'] = int(datetime.timestamp(datetime.now()))
            ckpt['args'] = vars(args)
            ckpt['step'] = itr + 1
            ckpt['lr'] = train_log_dict["lr"]
            torch.save(ckpt, os.path.join(save_path, 'ckpts', f'saved_model_{itr + 1}.pth'))

        if (itr + 1) % args.update_lr_itr == 0:
            lr = lr * args.update_lr_ratio
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            logging.info(f'update learning rate: {lr}')


def test_ood():
    for cate in ['boat', 'diningtable',
                 'motorbike', 'train', 'sofa', 'car']:
        args = parse_args()
        args.exp_name = f'1111_{cate}_ood_pseudo_10'
        args.category = cate
        args.iterations = 1000
        args.save_itr = 1000
        args.prev_ckpt = f'/home/jiahao/pretrain_6d_pose/experiments/1111_{cate}_ood_pseudo/ckpts/saved_model_1000.pth'
        if cate in ['bus', 'car', 'motorbike', 'aeroplane', 'bicycle', 'boat', 'diningtable', 'train']:
            args.distance = 6
        if cate in ['chair', 'bottle']:
            args.distance = 10
        if cate in ['sofa', 'tvmonitor']:
            args.distance = 8

        args.mesh_path = os.path.join(args.mesh_path, args.category, '01.off')
        args.train_img_path = os.path.join(args.dataset_path, 'train', 'images')
        args.train_anno_path = os.path.join(args.dataset_path, 'train', 'annotations')
        args.val_img_path = os.path.join(args.dataset_path, 'val', 'images')
        args.val_anno_path = os.path.join(args.dataset_path, 'val', 'annotations')
        args.val_img_path = None
        args.val_anno_path = None
        args.val = args.val_img_path is not None and args.val_anno_path is not None

        print(args)

        save_path = os.path.join(args.save_path, args.exp_name)
        ckpt_path = os.path.join(save_path, 'ckpts')

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(save_path, __file__))
        load_prev_ckpt = False
        if args.prev_ckpt is not None:
            load_prev_ckpt = True
            prev_ckpt = args.prev_ckpt

        logging.root.handlers = []
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(save_path, 'log.txt'),
                            filemode='a' if load_prev_ckpt else 'w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        logging.info(args)

        net = NetE2E(net_type=args.backbone, local_size=[args.local_size, args.local_size],
                     output_dimension=args.d_feature,
                     reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=True)
        logging.info(f'num params {sum(p.numel() for p in net.net.parameters())}')
        net = nn.DataParallel(net).cuda().train()
        if load_prev_ckpt:
            checkpoint = torch.load(prev_ckpt, map_location=args.device)
            net.load_state_dict(checkpoint['state'])
            print('load_prev_ckpt done')

        args.n = load_off(args.mesh_path)[0].shape[0]

        memory_bank = NearestMemoryManager(inputSize=args.d_feature,
                                           outputSize=args.n + args.num_noise * args.max_group,
                                           K=1, num_noise=args.num_noise, num_pos=args.n, momentum=args.adj_momentum)
        # memory_bank = memory_bank.cuda()
        if load_prev_ckpt:
            with torch.no_grad():
                # memory_bank.memory.copy_(checkpoint['memory'][0:memory_bank.memory.shape[0]])
                memory_bank.memory.copy_(checkpoint['memory'])
                print('load_memory done')

        memory_bank = memory_bank.cuda()

        criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_dataset, train_iterator, val_dataset, val_iterator = prepare_data_ood(args)
        logging.info(f'found {len(train_dataset)} for training ({args.category})')
        if args.val:
            logging.info(f'found {len(val_dataset)} for validation ({args.category})')

        # comet = Experiment(api_key='UKaGtn0xxEiEIPoG70XmgJhAB', project_name='nemo6d', workspace='wufeim')
        comet = None

        if comet:
            comet.set_name(args.exp_name)
            comet.log_parameters(vars(args))
            logging.info(comet.url)

        logging.info('Start training:')
        logging.info(f'Experiment:     {args.exp_name}')
        logging.info(f'Category:       {args.category}')
        logging.info(f'Num imgs train: {len(train_dataset)}')
        if args.val:
            logging.info(f'Num imgs val:   {len(val_dataset)}')
        logging.info(f'Total itr:      {args.iterations}')
        logging.info(f'LR:             {args.lr}')
        logging.info(f'Update LR itr:  {args.update_lr_itr}')
        logging.info(f'Updated LR:     {args.lr * args.update_lr_ratio}')

        log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
        log_val_loss, log_val_loss_main, log_val_loss_reg = [], [], []
        lr = args.lr

        for itr in range(args.iterations):
            train_log_dict = train_one_step(net, memory_bank, train_iterator, criterion, optim, args, itr)
            val_log_dict = val_one_step(net, memory_bank, val_iterator, criterion, args, itr)

            log_train_loss.append(train_log_dict['loss'])
            log_train_loss_main.append(train_log_dict['loss_main'])
            log_train_loss_reg.append(train_log_dict['loss_reg'])

            log_val_loss.append(val_log_dict['loss'])
            log_val_loss_main.append(val_log_dict['loss_main'])
            log_val_loss_reg.append(val_log_dict['loss_reg'])

            if itr == 0 or (itr + 1) % args.log_itr == 0:
                train_loss, train_loss_main, train_loss_reg = np.mean(log_train_loss), np.mean(
                    log_train_loss_main), np.mean(log_train_loss_reg)
                val_loss, val_loss_main, val_loss_reg = np.mean(log_val_loss), np.mean(log_val_loss_main), np.mean(
                    log_val_loss_reg)
                if itr > 0:
                    log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
                    log_val_loss, log_val_loss_main, log_val_loss_reg = [], [], []
                if comet:
                    comet.log_metric('lr', train_log_dict['lr'], step=itr + 1)
                    comet.log_metric('train_loss', train_loss, step=itr + 1)
                    comet.log_metric('train_loss_main', train_loss_main, step=itr + 1)
                    comet.log_metric('train_loss_reg', train_loss_reg, step=itr + 1)
                    comet.log_metric('val_loss', val_loss, step=itr + 1)
                    comet.log_metric('val_loss_main', val_loss_main, step=itr + 1)
                    comet.log_metric('val_loss_reg', val_loss_reg, step=itr + 1)

                logging.info(
                    f'[Itr {itr + 1}] lr={train_log_dict["lr"]} loss={train_loss:.5f} loss_main={train_loss_main:.5f} loss_reg={train_loss_reg:.5f}')

            if (itr + 1) % args.save_itr == 0:
                ckpt = {}
                ckpt['state'] = net.state_dict()
                ckpt['memory'] = memory_bank.memory
                ckpt['timestamp'] = int(datetime.timestamp(datetime.now()))
                ckpt['args'] = vars(args)
                ckpt['step'] = itr + 1
                ckpt['lr'] = train_log_dict["lr"]
                torch.save(ckpt, os.path.join(save_path, 'ckpts', f'saved_model_{itr + 1}.pth'))

            if (itr + 1) % args.update_lr_itr == 0:
                lr = lr * args.update_lr_ratio
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                logging.info(f'update learning rate: {lr}')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    if args.exp_type == 'pseudo':
        pseudo(args)
    if args.exp_type == 'fine_tune':
        fine_tune(args)
