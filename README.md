# Data-free Knowledge Distillation for Segmentation using Data-Enriching GAN

This repository is the official implementation of [Data-free Knowledge Distillation for Segmentation using Data-Enriching GAN
](https://arxiv.org/abs/2011.00809). 

### Deployment pipelines

![Banner](banner.png)

*(Top row)* A deployment pipeline where a large model is initially
trained on some dataset. Later for mobile deployment when the model needs to be compressed, the
dataset needs to be accessed again causing privacy concerns. 

*(Bottom row)* We present a deployment
pipeline where no access to data is required. Instead we use a proxy dataset to generate representative
samples to perform model compression.

### Data-Enriching GAN architecture

![Architecture](DeGAN.png)

## Requirements

To install requirements:

```setup
pip install pytorch-lightning==0.7.0
pip install tensorboard
```

## Training and evaluation

To train the DeGAN network

```train
python -m code.segmentation.degan_trainer
```
To train the student network

```train
python -m code.segmentation.datafree_kd_trainer
```


## Contributing

Create a issue or a pull request if you wish to contribute :) 
