import torch
from torch import nn
import torch.nn.functional as F

import models
import dataloaders
import argparse
import os
import datetime
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import data_aug
import utils








def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=100, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--normals', type=bool, default=False, help='If normals neeeded [default: False]')
    parser.add_argument('--shuffle', type=bool, default=True, help='If to be shuffled every epoch[default: True]')
    parser.add_argument('--mask', type=bool, default=False, help='wether mask to be used while training')

    parser.add_argument('--npoint', type=int,  default=1024, help='Point Number [changed to:8096]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args()




def main(args):


    


    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('../Results/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('decomposer_experiment')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, "decomposer_v1"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    TRAIN_DATASET = dataloaders.TraceParts_Dataloader(split='train',sample_points=args.npoint, normalize=True, cache_size=150,uniform=True,normals=args.normals,mask=args.mask)
    train_data = torch.utils.data.DataLoader(TRAIN_DATASET,batch_size=args.batch_size,shuffle=args.shuffle, num_workers=15,prefetch_factor=30)
    TEST_DATASET = dataloaders.TraceParts_Dataloader(split='val',sample_points=args.npoint, normalize=True, cache_size=150,uniform=True,normals=args.normals,mask=False)
    test_data = torch.utils.data.DataLoader(TEST_DATASET,batch_size=args.batch_size,shuffle=True, num_workers=20,prefetch_factor=40)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    if args.normals:
        channel = 6
    else:
        channel  = 3
    decomposer = models.ABDNet(input_dims=channel,if_avg_pool=False).cuda()
    criterion = models.loss()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        decomposer.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0



    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            decomposer.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = torch.optim.SGD(decomposer.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        decomposer = decomposer.apply(lambda x: bn_momentum_adjust(x,momentum))
        decomposer = decomposer.train()

        for (batch,data) in tqdm(enumerate(train_data),total=len(train_data),smoothing=0.9):
            if args.mask:
                points,target,mask = data
                mask = mask.float().cuda()
            else:
                points,target = data
                mask = None
            optimizer.zero_grad()
            points = points.data.numpy()
            points[:,:, 0:3] = data_aug.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = data_aug.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred = decomposer(points,mask=mask)
            seg_pred = seg_pred.contiguous().view(-1, 4)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            m_correct=[]
            for (batch,(points,target)) in tqdm(enumerate(test_data),total=len(test_data),smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                decomposer = decomposer.eval()
                seg_pred = decomposer(points,mask=None)
                
                seg_pred = seg_pred.contiguous().view(-1, 4)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                m_correct.append(correct.item() / (args.batch_size * args.npoint))
                test_loss = criterion(seg_pred,target)

            print(" Train loss :",loss.numpy(), "Test loss : ",test_loss.numpy())
            test_instance_acc = np.mean(m_correct)
            print('Validation accuracy is: %.5f' % test_instance_acc)
            log_string('Validation accuracy is: %.5f' % test_instance_acc)
            log_string(' Train loss : %.5f    Test loss : %.5f ' % (loss.numpy(),test_loss.numpy()))
        if test_instance_acc > best_acc:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_instance_acc,
                "train_loss":loss,
                "test_loss":test_loss,
                'model_state_dict': decomposer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            best_acc = test_instance_acc
        global_epoch+=1


if __name__ == '__main__':
    args = parse_args()
    utils.set_seed_globally()
    main(args)