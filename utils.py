#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Higher
@Email: hujiagao@gmail.com
"""
import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
from visdom import Visdom
from PIL import Image

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, exp_name='test', env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)

        if not os.path.exists('runs/%s/data/'%(exp_name)):
            os.makedirs('runs/%s/data/'%(exp_name))
        file = open('runs/%s/data/%s_%s_data.csv'%(exp_name, split_name, var_name), 'a')
        file.write('%d, %f\n'%(x, y))
        file.close()

    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )

    def plot_image(self, image, epoch, exp_name='test'):
        self.viz.image(image, env=exp_name+'_img', opts=dict(
            caption=epoch,
            ))

    def plot_images(self, images, run_split, epoch, nrow, padding=2, exp_name='test'):
        self.viz.images(images, env=exp_name+'_img', nrow=nrow, padding=padding, opts=dict(
            caption='%s_%d'%(run_split, epoch),
            # title='Random images',
            jpgquality=100,
            ))


def save_checkpoint(state, is_best, exp_name='test', filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    shutil.copyfile(filename, 'runs/%s/' % (exp_name) + 'model_newest.pth.tar')
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(exp_name) + 'model_best.pth.tar')


def cross_entropy_loss(predict, target, weight=None, size_average=True, ignore_index=-100):
    r"""
    :param predict: size (batch_size, n_class, h, w)
    :param target: size (batch_size, h, w)
    :param weight: a manual rescaling weight given to each
                class. If given, has to be a Tensor of size "n_class"
    :param size_average: By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: True
    :param ignore_index:Specifies a target value that is ignored
                and does not contribute to the input gradient. When size_average is
                True, the loss is averaged over non-ignored targets. Default: -100
    :return:
    """

    batch_size, n_class, h, w = predict.size()
    p_resize = predict.transpose(1, 2).transpose(2, 3).contiguous().view(-1, n_class) # size (batch_size*h*w, n_class)
    t_resize = target.view(-1,) # size (batch_size*h*w)
    return F.cross_entropy(p_resize, t_resize, weight=weight, size_average=size_average, ignore_index=ignore_index)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(base_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * ((1 - 0.015) ** epoch)
    if lr < 1e-4:
        lr = 1e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer.param_groups[0]['lr']

    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
      Input is numpy array
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    hist += np.finfo(float).eps  # in case of zero divided
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    f1 = 2 * np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1))
    fscore = np.nanmean(f1)

    return acc, acc_cls, mean_iu, fscore


# colour map
label_colours = [(0, 0, 0)
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                img[j, k] = label_colours[mask[i][j, k]]
        outputs[i] = img
    return outputs

def combine_label(labels, COMB_DICTs):
    """combine some label of segmentation masks for face dataset.
        Args:
          labels: finest-grained groundtruth label. torch tensor, size [n_batch, h, w]

        Returns:
          A batch of masks with combined labels. [grain_index, n, h, w]
        """
    n, h, w = labels.shape
    outputs = np.zeros((len(COMB_DICTs) + 1, n, h, w), dtype=np.uint8)
    for i in range(n):
        tmp = np.array(labels[i, :, :].numpy(), dtype=np.uint8)
        for j, combdict in enumerate(COMB_DICTs):
            comb_label = np.zeros((h, w), dtype=np.uint8)
            for key in combdict:
                comb_label[tmp == key] = combdict[key]

            outputs[j, i, :, :] = comb_label

        outputs[-1, i, :, :] = tmp

    return outputs