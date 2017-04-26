# Copyright (c) 2017 Shunta Saito

import os
import pickle

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer.links.caffe import CaffeFunction


class Illust2Vec(Chain):
    CAFFEMODEL_FN = 'illust2vec_ver200.caffemodel'
    PKL_FN = 'illust2vec_ver200.pkl'

    def __init__(self, n_classes):
        w = chainer.initializers.HeNormal()

        if not os.path.exists(self.PKL_FN):
            print('Converting Caffe model...')
            model = CaffeFunction(self.CAFFEMODEL_FN)
            pickle.dump(model, open(self.PKL_FN, 'wb'))
        else:
            model = pickle.load(open(self.PKL_FN, 'rb'))

        # Delete unused layers to save the memory consumption
        del model.encode1
        del model.encode2
        del model.forwards['encode1']
        del model.forwards['encode2']
        model._children.pop()
        model._children.pop()
        model.layers = model.layers[:-2]

        super(Illust2Vec, self).__init__(
            trunk=model,
            fc6=L.Linear(None, 4096, initialW=w),
            fc7=L.Linear(4096, 4096, initialW=w),
            fc8=L.Linear(4096, n_classes, initialW=w))
        self.train = True

    def __call__(self, x):
        h = self.trunk({'data': x}, ['conv6_3'], train=self.train)[0]
        h.unchain_backward()
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        return self.fc8(h)
