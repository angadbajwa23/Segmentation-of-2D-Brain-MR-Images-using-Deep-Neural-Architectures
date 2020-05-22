# Brain-Image-Segmentation
Brain tumor segmentation using a 3D UNet CNN

I used Keras with a Tensorflow backend. This UNet was built for the following dataset: https://figshare.com/articles/brain_tumor_dataset/1512427

3064 T1-weighted contrast-inhanced images with three kinds of brain tumor are provided in the dataset.

![datasete](./images/dataset.png)

The first half of the U-net is effectively a typical convolutional neural network like one would construct for an image classification task, with successive rounds of zero-padded ReLU-activated convolutions and ReLU-activated max-pooling layers. Instead of classification occurring at the "bottom" of the U, symmetrical upsampling and convolution layers are used to bring the pixel-wise prediction layer back to the original dimensions of the input image.

Here is the architecture for the 2D U-Net from the original publication mentioned earlier:

![u-net-architecture](./images/u-net-architecture.png)


Here's an example of the correlation between my predictions in a single 2D plane:

Ground Truth:               |  Prediction:
:-------------------------:|:-------------------------:
![ground truth](./images/Ground_Truth_Example.png)  |  ![prediction](./images/Prediction_Example.png)

Here's an example of the correlation between my predictions in a rotating 3D volume plane:

Ground Truth:               |  Prediction:
:-------------------------:|:-------------------------:
![ground truth](./images/groundtruth_rotation_example.gif)  |  ![prediction](./images/prediction_rotation_example.gif)

Reload the page or click each image to see the gif loop again.

The UNet was based on this paper: https://arxiv.org/abs/1802.10508

I heavily modified code from two sources to get this project to work:

- Original code for building the UNet was from this repo: https://github.com/ellisdg/3DUnetCNN

