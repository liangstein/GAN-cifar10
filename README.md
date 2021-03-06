## Dependency
* Python3.6(numpy, scipy, pickle, h5py, Pillow),
* Keras2.02,
* Tensorflow v1.1 backend, (Not Theano backend)
* Cuda8.0, Cudnn6.0 (If GPU is used)

## Neural Network Implementation
GAN is short for [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). GAN contains a generator and a discriminator, a generator tries to generate images that can cheat the discriminator, while the discriminator tries to distinguish the generated images from real images. In the training process, the discriminator should be better trained than the generator, hence the generator can learn from the discriminator. 

To make the generator powerful, BatchNormalization is usually used. However, using BatchNormalization makes the GAN un-stable. After many failures, we use Selu Activation function instead. What's more, one-sided label smoothing is used in training discriminator.

After training for 120 epochs, the generator can generate images that are similar to real photos. Let's see some pictures generated by GAN.
<p align="center">
  <img src="https://github.com/liangstein/GAN-cifar10/blob/master/presentation.png" width="400"/>
</p> 

## Authors
liangstein (lxxhlb@gmail.com) 

