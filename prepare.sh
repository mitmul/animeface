#!/usr/bin/env bash

if [ ! -f animeface-character-dataset.zip ]; then
    curl -O http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
fi
if [ ! -d animeface-character-dataset ]; then
    unzip animeface-character-dataset.zip
fi

if [ ! -f illust2vec_ver200.caffemodel ]; then
    curl -O http://illustration2vec.net/models/illust2vec_ver200.caffemodel
fi
if [ ! -f image_mean.npy ]; then
    curl -O http://illustration2vec.net/models/image_mean.npy
fi