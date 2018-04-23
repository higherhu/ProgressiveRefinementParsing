import argparse
import os
import sys
import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from cnns import SegNet, PSPNet, FCDenseNet
from image_loader_class import ImageLoader
from utils import VisdomLinePlotter, save_checkpoint, adjust_learning_rate, AverageMeter,cross_entropy_loss, \
    label_accuracy_score, decode_labels, get_1x_lr_params_NOscale, get_10x_lr_params

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')


parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--resume', type=str, default='', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_trainimgs', type=int, default=100000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument("--input-size", type=str, default='224,224',
                        help="Comma-separated string with height and width of images.")
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument("--dataset", type=str, default='HelenFace', 
                        help="Comma-separated string with height and width of images.")
parser.add_argument("--n_class", type=int, default=11, 
                     help="Number of classes to predict (including background).")

parser.set_defaults(test=False)
parser.set_defaults(learned=False)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=True)

best_mIoU = 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def unNormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    result = tensor.clone()
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)

    return result

def main():
    global args, best_mIoU
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.visdom:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.name+'_'+args.dataset)

    print(args.name + '_' + args.dataset, 'n_class: ', args.n_class)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(ImageLoader('../Dataset/', args.dataset, 'train.txt',
                                                           ignore_label=args.ignore_label, n_imgs=args.num_trainimgs,
                                                           crop_size=input_size,
                                                           transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               normalize,
                                                           ])),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(ImageLoader('../Dataset/', args.dataset, 'test.txt',
                                                          ignore_label=args.ignore_label, n_imgs=10000,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              normalize,
                                                          ])),
                                              batch_size=1, shuffle=True, **kwargs)

    net = _get_model_instance(args.name)(num_classes=args.n_class, pretrain=True, nIn=3)
    if args.cuda:
        net.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = cross_entropy_loss
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        print('Epoch: %d'%(args.start_epoch))
        test_acc, test_mIoU = test(val_loader, net, criterion, args.start_epoch, showall=True)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        lr = adjust_learning_rate(args.lr, optimizer, epoch)
        if args.visdom:
            plotter.plot('lr', 'learning rate', epoch, lr, exp_name=args.name+'_'+args.dataset)

        # train for one epoch
        cudnn.benchmark = True
        train(train_loader, net, criterion, optimizer, epoch)

        # evaluate on validation set
        cudnn.benchmark = False
        acc, mIoU = test(test_loader, net, criterion, epoch)

        # record best acc and save checkpoint
        is_best = mIoU > best_mIoU
        best_mIoU = max(mIoU, best_mIoU)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'best_mIoU': best_mIoU,
            'acc': acc
        }, is_best, exp_name=args.name+'_'+args.dataset, filename='checkpoint_%d.pth.tar'%(epoch))


def train(train_loader, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    mIoUs = AverageMeter()
    acc_clss = AverageMeter()
    fscores = AverageMeter()

    # switch to train mode
    net.train()
    start_time = time.time()
    all_start = start_time
    print('  Train Epoch  |      Loss    |      Acc       |     mIoU   |'
          '     Acc_cls    |    f-score     |  Time  ')
    for batch_idx, (datas, targets) in enumerate(train_loader):

        if args.cuda:
            datas, targets = datas.cuda(), targets.cuda()
        datas, targets = Variable(datas), Variable(targets)

        # compute output
        scores = net(datas)
        loss = criterion(scores, targets)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lbl_pred = scores.data.max(1)[1].cpu().numpy()[:, :, :] 
        lbl_true = targets.data.cpu().numpy()
        acc, acc_cls, mIoU, fscore = label_accuracy_score(lbl_true, lbl_pred, n_class=args.n_class)
        losses.update(loss.data[0], datas.size(0))
        accs.update(acc, datas.size(0))
        acc_clss.update(acc_cls, datas.size(0))
        mIoUs.update(mIoU, datas.size(0))
        fscores.update(fscore, datas.size(0))

        if batch_idx % args.log_interval == 0:
            duration = time.time() - start_time
            print('{:3d}[{:4d}/{:4d}] | {:.3f}({:.3f}) | {:2.2f}%({:2.2f}%) | {:.2f}({:.2f}) | {:2.2f}%({:2.2f}%) |'
                  ' {:2.2f}%({:2.2f}%) | ({:.3f} sec)'.format(
                epoch, batch_idx * len(datas), len(train_loader.dataset),
                losses.val, losses.avg, 100.*accs.val, 100.*accs.avg, mIoUs.val, mIoUs.avg,
                100.*acc_clss.val, 100.*acc_clss.avg, 100.*fscores.val, 100.*fscores.avg, duration))

            start_time = time.time()

    duration = time.time() - all_start
    print('Train Summary: Epoch {}, Acc: {:.2f}%, mIoU: {:.2f}, Acc_cls: {:.2f}%, f-score: {:.2f}% ({:.3f} sec)'.format(
        epoch, 100. * accs.avg, mIoUs.avg, 100 * acc_clss.avg, 100 * fscores.avg, duration))

    # log avg values to visdom
    if args.visdom:
        plotter.plot('acc', 'train', epoch, accs.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('loss', 'train', epoch, losses.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('mIoU', 'train', epoch, mIoUs.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('acc_cls', 'train', epoch, acc_clss.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('fscore', 'train', epoch, fscores.avg, exp_name=args.name+'_'+args.dataset)
        if epoch ==1 or epoch % 20 == 0:
            plot_images(datas, lbl_pred, lbl_true, epoch, split='train', crop_size=map(int, args.input_size.split(',')))


def test(test_loader, net, criterion, epoch, showall=False):
    losses = AverageMeter()
    accs = AverageMeter()
    mIoUs = AverageMeter()
    acc_clss = AverageMeter()
    fscores = AverageMeter()

    # switch to evaluation mode
    net.eval()
    start_time = time.time()
    for batch_idx, (datas, targets) in enumerate(test_loader):
        if args.cuda:
            datas, targets = datas.cuda(), targets.cuda()
        datas, targets = Variable(datas, volatile=True), Variable(targets, volatile=True) 

        # compute output
        scores = net(datas)
        testloss = criterion(scores, targets).data[0]

        # measure accuracy and record loss
        lbl_pred = scores.data.max(1)[1].cpu().numpy()[:, :, :] 
        lbl_true = targets.data.cpu().numpy()
        acc, acc_cls, mIoU, fscore = label_accuracy_score(lbl_true, lbl_pred, n_class=args.n_class)
        losses.update(testloss, datas.size(0))
        accs.update(acc, datas.size(0))
        acc_clss.update(acc_cls, datas.size(0))
        mIoUs.update(mIoU, datas.size(0))
        fscores.update(fscore, datas.size(0))

        if showall:
            plot_images(datas, lbl_pred, lbl_true, epoch, split='test', crop_size=map(int, args.input_size.split(',')))

    duration = time.time() - start_time
    print('\nTest set: Loss: {:.4f}, Acc: {:.2f}%, mIoU: {:.4f}, Acc_cls: {:.2f}%, f-score: {:.2f}% ({:.3f} sec)\n'.format(
        losses.avg, 100.*accs.avg, mIoUs.avg, 100*acc_clss.avg, 100*fscores.avg, duration))
    if args.visdom:
        plotter.plot('acc', 'test', epoch, accs.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('loss', 'test', epoch, losses.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('mIoU', 'test', epoch, mIoUs.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('acc_cls', 'test', epoch, acc_clss.avg, exp_name=args.name+'_'+args.dataset)
        plotter.plot('fscore', 'test', epoch, fscores.avg, exp_name=args.name+'_'+args.dataset)

        # plot images in a grid
        if epoch == 1 or epoch % 20 == 0:
            plot_images(datas, lbl_pred, lbl_true, epoch, split='test', crop_size=map(int, args.input_size.split(',')))

    return accs.avg, mIoUs.avg


def plot_images(imdata, lbl_pred, lbl_true, epoch, split='test', crop_size=None):
    img = unNormalize(imdata.data).cpu().numpy()[0] * 255
    img = img.astype(np.uint8)  # (c, h, w)
    pred = decode_labels(lbl_pred, num_images=1, num_classes=args.n_class)[0]  # (h, w, c)
    gt = decode_labels(lbl_true, num_images=1, num_classes=args.n_class)[0]
    pred = np.moveaxis(pred, 2, 0)  # (c, h, w)
    gt = np.moveaxis(gt, 2, 0)

    if crop_size is not None: # center crop
        _, h, w = img.shape
        tx, ty, bx, by = 0, 0, w, h
        ch, cw = crop_size
        if ch > h:
            ty, by = 0, 0
        else:
            ty, by = (h-ch)//2, h-(h-ch)//2
        if cw > w:
            tx, bx = 0, 0
        else:
            tx, bx = (w-cw)//2, w-(w-cw)//2
        img = img[:, ty:by, tx:bx]
        pred = pred[:, ty:by, tx:bx]
        gt = gt[:, ty:by, tx:bx]

    grid_imgs = [img, pred, gt]
    plotter.plot_images(grid_imgs, split, epoch, nrow=3, exp_name=args.name + '_' + args.dataset)


def _get_model_instance(name):
    return {
        'SegNet': SegNet,
        'PSPNet': PSPNet,
        'FCDenseNet': FCDenseNet
    }[name]


if __name__=='__main__':
    main()
