# animeface

This repository is a part of Chainer tutorial handouts. It shows

- How to write your original dataset class in Chainer
- How to fine-tune a pre-trained model on another dataset

## Requirement

- Chainer 1.23.0+
- PIL 4.0.0+
- NumPy 1.12.0+

## Prepare dataset

Just run

```bash
bash prepare.sh
```

## Start training

```bash
python train.py --gpus 0 --batchsize 128
```

It takes about 30 minutes to finish the training on Pascal TITAN X (12GB memory).

## Result

The accuracy on validation dataset after 120 epochs is 0.916953.

### Loss plot

![](https://raw.githubusercontent.com/mitmul/animeface/wiki/images/loss.png)

### Accuracy plot

![](https://raw.githubusercontent.com/mitmul/animeface/wiki/images/accuracy.png)