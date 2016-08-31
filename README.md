# dsg-qualification

[<img src="http://www.datasciencegame.com/static/images/logo.png" height="40">](http://www.datasciencegame.com/)
[<img src="http://www.uva-nemo.org/images/uva_logo_40.png" height="40">](http://www.english.uva.nl/)

Data Science Game Competition Qualification Round Solution by The Nerd Herd Team of the University of Amsterdam.


## Dependencies

* Theano
* Lasagne
* Keras
* Chainer
* cupy
* numpy

Each of us liked different frameworks, sorry for that :)

## Configuring

Create a directory **data** inside the project directory and put *sample_submission4.csv*, *id_train.csv* files inside. All images should be in the directory **data/roof_images/**.

## Pretrained Models

Our solution uses pretrained on ImageNet models of [GoogleNet](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl) ([Going Deeper with Convolutions](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) by Szegedy et al.) and [Inception V3](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl) ([Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/pdf/1512.00567v3.pdf) by Szegedy et al.) available in [Lasagne](https://github.com/Lasagne/Recipes/tree/master/modelzoo) and [VGG16](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) and [VGG19](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d) models ([Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/pdf/1409.1556.pdf) by Simonyan and Zisserman) available for [Keras](http://keras.io/) to apply transfer learning ([Rich feature hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/pdf/1311.2524v5) by Girshick et al.) for the roof direction classification task.

## Architecture

We use a modification of the ResNet in combination with state-of-the art-based models (GoogleNet, InceptionV3, VGG16, VGG19) on multiple image scales (32x32, 64x64, 80x80, 96x96, 104x104) initialized with pretrained models to build a single bagged model with majority voting strategy. L2 normalization is applied on randomly initialized layers. Following models have been included into the best majority voting model:

Architecture | Image Size | Validation Accuracy | Batch Size | Learning Rate | Dropout Ratio | Test Accuracy (40%)
-------------|------------|---------------------|------------|---------------|---------------|--------------------
GoogleNet    | 64x64      | 0.810               | 26         | 0.0073725     | 0.17844       | 0.82675
GoogleNet    | 64x64      | 0.802               | 37         | 0.0051616     | 0.36257       | 0.80157
GoogleNet    | 80x80      | 0.811               |	22         | 0.0059465	   | 0.26172       | N/A
GoogleNet    | 96x96      | 0.812               |	34	       | 0.0061347     | 0.30162       | 0.81907
InceptionV3  | 64x64      | 0.810               | 31	       | 0.007212      |               | 0.81836
InceptionV3  | 80x80      | 0.819	              | 25	       | 0.0031446     |               | N/A
InceptionV3  | 80x80      | 0.820               |	39	       | 0.0072336     |               | N/A
InceptionV3  | 96x96      | 0.824	              | 28	       | 0.0060929	   |               | 0.8264
InceptionV3  | 96x96      | 0.838               |	20         | 0.0042952	   |               | 0.83122
VGG 16       | 80x80      | 0.826               | N/A        | N/A           | N/A           | N/A
VGG 16       | 96x96      | 0.820               | N/A        | N/A           | N/A           | N/A
VGG 16       | 64x64      | 0.816               | N/A        | N/A           | N/A           | 0.82443
VGG 16       | 80x80      | 0.806               | N/A        | N/A           | N/A           | N/A
VGG 19       | 112x112    | 0.818               | N/A        | N/A           | N/A           | 0.80800
ResNet       | 64x64      | 0.805               | 128        | 0.007         | 0.004         | 0.81532
ResNet       | 32x32      | 0.794               | 128        | 0.007         | 0.004         | 0.78532

<sup>N/A for the VGG models is due to the hyperparameters not correctly being saved. N/A in the test data column is due to not all models being handed in.</sup>


## Parameters tuning

Our models implement a hyperparameters tuning approach described in the paper [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) by James Bergstra and Yoshua Bengio and use validation set for early stopping ([Overfitting in Neural Nets: Backpropagation, Conjugate Gradient, and Early Stopping](https://www.semanticscholar.org/paper/Overfitting-in-Neural-Nets-Backpropagation-Caruana-Lawrence/072d756c8b17a78018298e67ff29e6d3a4fe5770/pdf) by Caruana et al.) to prevent overfitting. For the validation data we picked last 1000 images of the provided annotated dataset.

### GoogleNet and InceptionV3 tuning
To tune GoogleNet models run

```
python -u main.py --model GoogleNet --train --tune --version v1 --img_dim <image dimension>
```

To tune InceptionV3 models run

```
python -u main.py --model GoogleNet --train --tune --version v3 --img_dim <image dimension>
```

A weight file model.npz will be generated with a best found accuracy on the validation set. Submission file can be generated further by running:

```
python main.py --model GoogleNet --version v3 --img_dim <image dimension> --name <submission prefix> --weights model.npz
```

This script will generate a validation internal meta csv file in the **submissions_val** directory and a submission file in the directory **submissions** with a proper file name suffices ``submission prefix``.

### VGG16 and VGG19 tuning

We have used pretrained VGG models, which were developed by [Simonyan and Zisserman](https://arxiv.org/abs/1409.1556). The VGG 16 model was taken from [this](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3), the weights can be downloaded [here](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view). The VGG 19 model can be obtained from this [link](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d) and its respective weights from [this link](https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view).

If you wish to train the vgg models, download their weights and place them in shesame folder main.py is located!

In order to tune the VGG 16 model type the following into the command line:
```
python -u main.py --model VGG --train --tune --version 16 --img_dim <image dimension>
```

Similarly, in order to tune the VGG 19 model use this command:

```
python -u main.py --model VGG --train --tune --version 19 --img_dim <image dimension>
```
After tuning you may generate a submission file for the vgg 16 and 19 respectively,like so:
```
python main.py --model VGG --version 19 --img_dim <image dimension>
```
```
python main.py --model VGG --version 16 --img_dim <image dimension>
```

Please ensure that image dimensions remain the same for tuning and predicting.

### ResNet tuning

Train a ResNet model with the command:
```
python main.py —-model ResNet —-train
```

To create submission csv files run:
```
python main.py -—model ResNet
```

To recreate the 32x32 model, instead of import models.ResNet, one should import models.ResNet_32 and change the input images size. The commands will be the same as described above.

## Bagging/Making a voting submission

In our research we used three different voting strategies; majority voting, majority voting weighted by model accuracy and majority voting weighted by model per class accuracy. The accuracies were determined on our 1000 image validation set but this also provided the problem that came with applying weights on the votes based on the validation set. The bagging approach would overfit on the validation set, therefore we just used normal majority voting in the end.

To make a final bagged submission  make sure all submission files from the separate models are in the **submissions** folder and run:
```
python make_prediction_file.py
```

It will produce a file called **submission_vote.csv** in the **submission** folder.

## Copyrights

Copyright (c) 2016 Minh Ngo
Copyright (c) 2016 Riaan Zoetmulder
Copyright (c) 2016 Emiel Hoogeboom
Copyright (c) 2016 Wolf Vos

The source code is distributed under the term of the [MIT License](LICENSE).
