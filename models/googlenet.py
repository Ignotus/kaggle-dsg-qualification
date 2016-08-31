from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np

import os
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import DropoutLayer
try:
  from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
  from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
except ImportError:
  from lasagne.layers import Conv2DLayer as ConvLayer
  from lasagne.layers import MaxPool2DLayer as PoolLayerDNN

from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear

from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers.normalization import batch_norm


def print_progress(percentage, loss, acc, final=False):
    slashes = int(percentage * 25)
    print('\r[' + ''.join(['#' for i in range(slashes)]) + ''.join([' ' for i in range(25 - slashes)]) + '] %.2f %%, loss %.3f, acc %.3f' % (percentage * 100, loss, acc), end='')
    if final:
        print()

def profile(func):
    import time
    def inner(*args, **kwargs):
        time1 = time.time()
        result = func(*args, **kwargs)
        print('Took %d seconds' % int(time.time() - time1))
        return result
    return inner

class GoogleNet(object):
    def __init__(self, weight_file=None, forward=False, learning_rate=0.001, dropout=0.4, lamb=0.00001):
        self.input_var = T.tensor4('inputs')

        self.net = self.build_model(self.input_var, forward, dropout)

        if weight_file is not None:
            self.load_weights(weight_file)

        prediction = lasagne.layers.get_output(self.net['prob'])

        self.target_var = T.ivector('targets')
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)
        loss = loss.mean() + lamb * lasagne.regularization.l2(self.net['prob'].W)

        params = lasagne.layers.get_all_params(self.net['prob'], trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate)

        test_prediction = lasagne.layers.get_output(self.net['prob'], deterministic=True)

        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)

        self.train_fn = theano.function([self.input_var, self.target_var], [loss, test_acc], updates=updates)
        self.val_fn = theano.function([self.input_var, self.target_var], [loss, test_acc])
        self.predict_fn = theano.function([self.input_var], [T.argmax(test_prediction, axis=1)])

    @profile
    def train_epoch(self, data, labels, batch_size=64):
        print('Training')
        loss = 0.0
        acc = []
        for i in range(0, len(labels), batch_size):
            train_loss, train_acc = self.train_fn(data[i:i + batch_size], labels[i:i + batch_size])
            print_progress(i / float(len(labels)), train_loss, train_acc)
            acc.append(train_acc)
            loss += train_loss
        acc = np.mean(acc)

        print_progress(1., loss, acc, final=True)
        return loss, acc

    @profile
    def eval(self, data, labels, batch_size=128):
        print('Evaluating')
        acc = []
        loss = 0.0
        for i in range(0, len(labels), batch_size):
            test_loss_val, test_acc_val = self.val_fn(data[i:i + batch_size], labels[i:i + batch_size])
            print_progress(i / float(len(labels)), test_loss_val, test_acc_val)
            loss += test_loss_val
            acc.append(test_acc_val)
        acc = np.mean(acc)
        print_progress(1.0, loss, acc, final=True)
        return loss, acc

    @profile
    def predict(self, data, batch_size=128):
        print('Predicting')
        predictions = list()
        for i in range(0, len(data), batch_size):
            prediction = self.predict_fn(data[i:i + batch_size])[0]
            print_progress(i / float(len(data)), 0, 0)
            predictions.append(prediction)
        return np.hstack(predictions)

    def write(self, file_name):
        np.savez(file_name, *lasagne.layers.get_all_param_values(self.net['prob']))

    def read(self, file_name):
        weights = np.load(open(file_name))
        param_values = [weights['arr_%d' % i] for i in range(len(weights.files))]
        lasagne.layers.set_all_param_values(self.net['prob'], param_values)

    def load_weights(self, weight_file):
        import pickle
        model = pickle.load(open(weight_file))
        lasagne.layers.set_all_param_values(self.net['dropout1'], model['param values'][:-2])

    def build_inception_module(self, name, input_layer, nfilters):
        # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
        net = dict()
        net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
        net['pool_proj'] = ConvLayer(
            net['pool'], nfilters[0], 1, flip_filters=False)

        net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

        net['3x3_reduce'] = ConvLayer(
            input_layer, nfilters[2], 1, flip_filters=False)
        net['3x3'] = ConvLayer(
            net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

        net['5x5_reduce'] = ConvLayer(
            input_layer, nfilters[4], 1, flip_filters=False)
        net['5x5'] = ConvLayer(
            net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

        net['output'] = ConcatLayer([
            net['1x1'],
            net['3x3'],
            net['5x5'],
            net['pool_proj'],
            ])

        return {'{}/{}'.format(name, k): v for k, v in net.items()}

    def build_model(self, input_var, forward, dropout):
        net = dict()
        net['input'] = InputLayer((None, 3, None, None), input_var=input_var)
        net['conv1/7x7_s2'] = ConvLayer(
            net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
        net['pool1/3x3_s2'] = PoolLayer(
            net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
        net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
        net['conv2/3x3_reduce'] = ConvLayer(
            net['pool1/norm1'], 64, 1, flip_filters=False)
        net['conv2/3x3'] = ConvLayer(
            net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
        net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
        net['pool2/3x3_s2'] = PoolLayerDNN(net['conv2/norm2'], pool_size=3, stride=2)

        net.update(self.build_inception_module('inception_3a',
                                               net['pool2/3x3_s2'],
                                               [32, 64, 96, 128, 16, 32]))
        net.update(self.build_inception_module('inception_3b',
                                               net['inception_3a/output'],
                                               [64, 128, 128, 192, 32, 96]))
        net['pool3/3x3_s2'] = PoolLayerDNN(net['inception_3b/output'],
                                           pool_size=3, stride=2)

        net.update(self.build_inception_module('inception_4a',
                                               net['pool3/3x3_s2'],
                                               [64, 192, 96, 208, 16, 48]))
        net.update(self.build_inception_module('inception_4b',
                                               net['inception_4a/output'],
                                               [64, 160, 112, 224, 24, 64]))
        net.update(self.build_inception_module('inception_4c',
                                               net['inception_4b/output'],
                                               [64, 128, 128, 256, 24, 64]))
        net.update(self.build_inception_module('inception_4d',
                                               net['inception_4c/output'],
                                               [64, 112, 144, 288, 32, 64]))
        net.update(self.build_inception_module('inception_4e',
                                               net['inception_4d/output'],
                                               [128, 256, 160, 320, 32, 128]))
        net['pool4/3x3_s2'] = PoolLayerDNN(net['inception_4e/output'],
                                           pool_size=3, stride=2)

        net.update(self.build_inception_module('inception_5a',
                                               net['pool4/3x3_s2'],
                                               [128, 256, 160, 320, 32, 128]))
        net.update(self.build_inception_module('inception_5b',
                                               net['inception_5a/output'],
                                               [128, 384, 192, 384, 48, 128]))

        net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])

        if forward:
            #net['fc6'] = DenseLayer(net['pool5/7x7_s1'], num_units=1000)
            net['prob'] = DenseLayer(net['pool5/7x7_s1'], num_units=4, nonlinearity=softmax)
        else:
            net['dropout1'] = DropoutLayer(net['pool5/7x7_s1'], p=dropout)
            #net['fc6'] = DenseLayer(net['dropout1'], num_units=1000)
            #net['dropout2'] = DropoutLayer(net['fc6'], p=dropout)
            net['prob'] = DenseLayer(net['dropout1'], num_units=4, nonlinearity=softmax)
        return net


class InceptionV3(GoogleNet):
    def __init__(self, weight_file=None, forward=False, learning_rate=0.001, dropout=0.4, lamb=0.00001):
        super(InceptionV3, self).__init__(weight_file, forward, learning_rate, dropout, lamb)

    def bn_conv(self, input_layer, **kwargs):
        l = Conv2DLayer(input_layer, **kwargs)
        l = batch_norm(l, epsilon=0.001)
        return l

    def inceptionA(self, input_layer, nfilt):
        # Corresponds to a modified version of figure 5 in the paper
        l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

        l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

        l3 = self.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
        l3 = self.bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
        l3 = self.bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

        l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
        l4 = self.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

        return ConcatLayer([l1, l2, l3, l4])


    def inceptionB(self, input_layer, nfilt):
        # Corresponds to a modified version of figure 10 in the paper
        l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

        l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
        l2 = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

        l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

        return ConcatLayer([l1, l2, l3])


    def inceptionC(self, input_layer, nfilt):
        # Corresponds to figure 6 in the paper
        l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

        l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
        l2 = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

        l3 = self.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
        l3 = self.bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
        l3 = self.bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
        l3 = self.bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
        l3 = self.bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

        l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
        l4 = self.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

        return ConcatLayer([l1, l2, l3, l4])

    def inceptionD(self, input_layer, nfilt):
        # Corresponds to a modified version of figure 10 in the paper
        l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
        l1 = self.bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

        l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
        l2 = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
        l2 = self.bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

        l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

        return ConcatLayer([l1, l2, l3])


    def inceptionE(self, input_layer, nfilt, pool_mode):
        # Corresponds to figure 7 in the paper
        l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

        l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2a = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
        l2b = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

        l3 = self.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
        l3 = self.bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
        l3a = self.bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
        l3b = self.bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

        l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

        l4 = self.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

        return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])

    def load_weights(self, weight_file):
        import pickle
        model = pickle.load(open(weight_file))

        lasagne.layers.set_all_param_values(self.net['pool3'], model['param values'][:350])

    def build_model(self, input_var, forward, dropout):
        net = dict()

        net['input'] = InputLayer((None, 3, None, None), input_var=input_var)
        net['conv'] = self.bn_conv(net['input'],
                              num_filters=32, filter_size=3, stride=2)
        net['conv_1'] = self.bn_conv(net['conv'], num_filters=32, filter_size=3)
        net['conv_2'] = self.bn_conv(net['conv_1'],
                                num_filters=64, filter_size=3, pad=1)
        net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

        net['conv_3'] = self.bn_conv(net['pool'], num_filters=80, filter_size=1)

        net['conv_4'] = self.bn_conv(net['conv_3'], num_filters=192, filter_size=3)

        net['pool_1'] = Pool2DLayer(net['conv_4'],
                                    pool_size=3, stride=2, mode='max')
        net['mixed/join'] = self.inceptionA(
            net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
        net['mixed_1/join'] = self.inceptionA(
            net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

        net['mixed_2/join'] = self.inceptionA(
            net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

        net['mixed_3/join'] = self.inceptionB(
            net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

        net['mixed_4/join'] = self.inceptionC(
            net['mixed_3/join'],
            nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

        net['mixed_5/join'] = self.inceptionC(
            net['mixed_4/join'],
            nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

        net['mixed_6/join'] = self.inceptionC(
            net['mixed_5/join'],
            nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

        net['mixed_7/join'] = self.inceptionC(
            net['mixed_6/join'],
            nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

        # net['mixed_8/join'] = self.inceptionD(
        #     net['mixed_7/join'],
        #     nfilt=((192, 320), (192, 192, 192, 192)))

        # net['mixed_9/join'] = self.inceptionE(
        #     net['mixed_8/join'],
        #     nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        #     pool_mode='average_exc_pad')

        # net['mixed_10/join'] = self.inceptionE(
        #     net['mixed_9/join'],
        #     nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        #     pool_mode='max')

        net['pool3'] = GlobalPoolLayer(net['mixed_7/join'])

        net['prob'] = DenseLayer(
            net['pool3'], num_units=4, nonlinearity=softmax)

        return net


def train(train_data, train_labels,
          valid_data, valid_labels,
          model='v1',
          learning_rate=0.0005, dropout=0.4, batch_size=64, lamb=0.00001, epochs=30):
    if model == 'v1':
        model = GoogleNet(learning_rate=learning_rate, dropout=dropout)

        # Pretrained GoogleNet model on ImageNet 
        # Taken from here https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl
        # License: Unrestricted Use
        # https://github.com/Lasagne/Recipes/blob/master/modelzoo/googlenet.py
        model.load_weights('/var/node436/local/mngo/src/facedu/Models/GoogleNet-224/blvc_googlenet.pkl')
    else:
        model = InceptionV3(learning_rate=learning_rate, dropout=dropout)

        # Pretrained GoogleNet model on ImageNet 
        # Taken from here https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl
        # License: Unrestricted Use
        # https://github.com/Lasagne/Recipes/blob/master/modelzoo/inception_v3.py
        model.load_weights('/var/node436/local/mngo/src/facedu/Models/GoogleNet-224/inception_v3.pkl')

    prev_valid_accuracy = 0
    wait = 0
    for epoch in range(1, epochs + 1):
        print('::: Epoch %d :::' % epoch)
        indexes = np.random.choice(len(train_labels), len(train_labels))
        train_data = train_data[indexes]
        train_labels = train_labels[indexes]

        train_loss, train_acc = model.train_epoch(train_data, train_labels, batch_size=batch_size)

        loss, acc = model.eval(valid_data, valid_labels)
        print('Validation loss %.3f, accuracy %.3f' % (loss, acc))

        if train_acc > 0.99: # Overfits
            break
        if acc < prev_valid_accuracy:
            wait += 1
            if wait > 2:
                break
        else:
            wait = 0
            prev_valid_accuracy = acc
            print('Storing a model')
            model.write('model_tmp.npz')

    return prev_valid_accuracy

def predict(valid_data, valid_labels, test_data, model='v1', model_path='model.npz'):
    if model == 'v1':
        model = GoogleNet(forward=True)
    else:
        model = InceptionV3(forward=True)
    model.read(model_path)

    loss, acc = model.eval(valid_data, valid_labels)
    print('Validation loss %.3f, accuracy %.3f' % (loss, acc))

    prediction = model.predict(test_data)
    prediction += 1
    return prediction, model.predict(valid_data) + 1

def random_search_hyperparameters(train_data, train_labels,
                                  valid_data, valid_labels, model='v1', times=40):
    """
    Implementation of the paper http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
    """
    import random
    print('Tuning')
    ### PARAMETERS TUNING CODE ###
    best_BATCH = 0
    best_LR = 0
    best_DO = 0
    best_acc = 0
    best_L2 = 0

    for i in range(times):
        BATCH = random.randint(20, 40)
        LR = random.uniform(0.0025, 0.0075)
        if model == 'v1':
            DO = random.uniform(0.15, 0.45)
        else:
            DO = 0.5
        L2 = 0.00001
        print('Trying BATCH %d, LR %.7f, DO %.5f, L2 %.6f' % (BATCH, LR, DO, L2))
        acc = train(train_data, train_labels, valid_data, valid_labels, model, LR, DO, BATCH, L2)
        if acc > best_acc:
            best_acc = acc
            best_BATCH = BATCH
            best_LR = LR
            best_DO = DO
            best_L2 = L2
            print('Best one updated!')
            os.rename('model_tmp.npz', 'model.npz')
    print('BEST BATCH %d, LR %.7f, DO %.5f, ACC %.5f' % (best_BATCH, best_LR, best_DO, best_acc))
    return best_BATCH, best_LR, best_DO, best_acc
