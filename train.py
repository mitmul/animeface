#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Shunta Saito

import argparse

import chainer
import chainer.links as L
from chainer import datasets
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from dataset import AnimeFaceDataset
from model import Illust2Vec

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, nargs='*', default=-1)
parser.add_argument('--batchsize', type=int, default=128)
args = parser.parse_args()

# Prepare dataset
d = AnimeFaceDataset()
train, valid = datasets.split_dataset_random(d, int(len(d) * 0.75), seed=0)

# Prepare iterator
train_iter = iterators.MultiprocessIterator(train, args.batchsize)
valid_iter = iterators.MultiprocessIterator(
    valid, args.batchsize, repeat=False, shuffle=False)

# Prepare model
model = Illust2Vec(len(d.cls_labels))
model = L.Classifier(model)

# Prepare optimizer
optimizer = optimizers.MomentumSGD(lr=0.01)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# Prepare devices
devices = {'main': args.gpus[0]}
for gid in args.gpus:
    devices.update({'gpu{}'.format(gid): gid})

# Prepare Updater
if len(args.gpus) == 1:
    updater = training.StandardUpdater(
        train_iter, optimizer, device=devices['main'])
else:
    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)


class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def create_lr_drop(drop_ratio=0.1):
    @training.make_extension()
    def lr_drop(trainer):
        trainer.updater.get_optimizer('main').lr *= drop_ratio

    return lr_drop


# Prepare trainer
trainer = training.Trainer(updater, (120, 'epoch'), out='AnimeFace-result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.observe_lr())

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy',
     'elapsed_time', 'lr']))

trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'epoch', file_name='loss.png'))

trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'epoch', file_name='accuracy.png'))

trainer.extend(TestModeEvaluator(valid_iter, model, device=gpu_id))
trainer.extend(create_lr_drop(drop_ratio=0.1), trigger=(40, 'epoch'))
trainer.run()
