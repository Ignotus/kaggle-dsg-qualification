import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from progressbar import ProgressBar
import cupy as cp
import os
import inspect
import shutil
import imp

class ResBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, fiber_map='id',
                 stride=1, pad=1, wscale=1, bias=0, nobias=False, use_cudnn=True, initialW=None, initial_bias=None):

        assert ksize % 2 == 1

        assert pad == (ksize - 1) // 2

        super(ResBlock, self).__init__(
            bn1=L.BatchNormalization(in_channels),
            conv1=L.Convolution2D(in_channels, out_channels, ksize, stride, pad, wscale),
            bn2=L.BatchNormalization(out_channels),
            conv2=L.Convolution2D(out_channels, out_channels, ksize, 1, pad, wscale),
        )
        if fiber_map == 'id':
            assert in_channels == out_channels
            self.fiber_map = F.identity
        elif fiber_map == 'linear':
            self.add_link('fiber_map', L.Convolution2D(in_channels, out_channels, 1, 2, 0, wscale))
        else:
            raise ValueError('Unimplemented fiber map {}'.format(fiber_map))


    def __call__(self, x, train, finetune):
        h = self.conv1(F.relu(self.bn1(x, test=not train, finetune=finetune)))
        h = self.conv2(F.relu(self.bn2(h, test=not train, finetune=finetune)))

        hx = self.fiber_map(x)
        return hx + h


class ResNet(chainer.ChainList):
    def __init__(self, num_blocks=3, nc32=8, nc16=16, nc8=32, dropout=0.5):
        ksize = 3
        pad = 1
        wscale = np.sqrt(2)
        self.dropout = dropout

        super(ResNet, self, ).__init__()
        
        self.add_link(
            L.Convolution2D(in_channels=3, out_channels=nc32, ksize=ksize, stride=1, pad=pad, wscale=wscale)
        )

        for i in range(num_blocks):
            self.add_link(
                ResBlock(
                    in_channels=nc32, out_channels=nc32, ksize=ksize,
                    fiber_map='id', stride=1, pad=pad, wscale=wscale
                )
            )

        for i in range(num_blocks):
            in_channels = nc16 if i > 0 else nc32
            fiber_map = 'id' if i > 0 else 'linear'
            stride = 1 if i > 0 else 2

            self.add_link(
                ResBlock(
                    in_channels=in_channels, out_channels=nc16, ksize=ksize,
                    fiber_map=fiber_map, stride=stride, pad=pad, wscale=wscale
                )
            )

        for i in range(num_blocks):
            in_channels = nc8 if i > 0 else nc16
            fiber_map = 'id' if i > 0 else 'linear'
            stride = 1 if i > 0 else 2

            self.add_link(
                ResBlock(
                    in_channels=in_channels, out_channels=nc8, ksize=ksize,
                    fiber_map=fiber_map, stride=stride, pad=pad, wscale=wscale
                )
            )
        self.add_link(
            F.BatchNormalization(nc8)
            )
        self.add_link(L.Convolution2D(in_channels=nc8, out_channels=5, ksize=4, wscale=wscale))
        
    def h(self, x, train, finetune):
        # First convolution layer.
        h = self[0](x)

        h = F.dropout(h, ratio=self.dropout, train=train)

        # Residual blocks.
        for i in range(1, len(self) - 2):
            h = self[i](h, train, finetune)

        # Batch normalization.
        h = self[-2](h, test=not train, finetune=finetune)
        h = F.relu(h)

        # Average pooling.
        h = F.max_pooling_2d(h, ksize=2, pad=0)

        # Prediction layer 5.
        h = self[-1](h)
        h = F.reshape(h, (h.data.shape[0], 5))

        return h

    def __call__(self, x, t, train=True, finetune=False):
        h = self.h(x, train, finetune)

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def predict(self, x):     
        h = self.h(x, train=False, finetune=False)
        return h.data.argmax(axis=1)

    def start_finetuning(self):
        for c in self.children():
            if isinstance(c, F.BatchNormalization):
                c.start_finetuning()

            if isinstance(c, ResBlock):
                c.bn1.start_finetuning()
                c.bn2.start_finetuning()


def do_epoch(data, labels, model, optimizer, batchsize, train=True, finetune=False, gpu=0):
    N = data.shape[0]
    pbar = ProgressBar(0, N)

    perm = np.random.permutation(N)
    sum_accuracy = 0.
    sum_loss = 0.

    for i in range(0, N, batchsize):
        x_batch = data[perm[i:i+batchsize]]
        t_batch = labels[perm[i:i+batchsize]]

        if gpu >= 0:
            x_var = Variable(cuda.to_gpu(x_batch))
            t_var = Variable(cuda.to_gpu(t_batch))
        else:
            x_var = Variable(x_batch)
            t_var = Variable(t_batch)

        if train:
            model.zerograds()

        loss, accuracy = model(x_var, t_var, train, finetune)

        if train:
            loss.backward()
            optimizer.update()

        sum_loss += loss.data * x_batch.shape[0]
        sum_accuracy += accuracy.data * x_batch.shape[0]

        pbar.update(i + x_batch.shape[0])

    return sum_loss, sum_accuracy

def setup_model(num_blocks, dropout, optimizer_class):
    model = ResNet(num_blocks=num_blocks, dropout=dropout)
    optimizer = optimizer_class()
    optimizer.setup(model)
    return model, optimizer

def random_search(train_data, train_labels, val_data, val_labels):
    print 'Randomly searching hyperparameters...' 

    if not os.path.exists('results/'):
            os.mkdir('results')

    for i in range(40):
        dropout=np.random.uniform(0.5, 0.7)
        num_blocks=np.random.randint(2, 4)
        lr = np.random.uniform(0.001, 0.01)
        weight_decay = np.random.uniform(0.001, 0.01)

        result_dir = os.path.join('results', 'dropout{:.2f}_blocks{:d}_lr{:.3f}_wd{:.3f}'.format(dropout,
                                                                                                 num_blocks,
                                                                                                 lr,
                                                                                                 weight_decay))

        train(train_data, train_labels, val_data, val_labels, epochs=40, batchsize=128, valid_freq=1, num_blocks=num_blocks, dropout=dropout,
          gpu=0, weight_decay=weight_decay, lr=lr, lr_schedule=(25, 50, 75), lr_decay=0.1, patience=4, result_dir=result_dir)





def train(train_data, train_labels, val_data, val_labels, epochs=40, batchsize=128, valid_freq=1, num_blocks=3, dropout=0.5,
          gpu=0, weight_decay=0.001, lr=0.005, lr_schedule=(25, 50, 75), lr_decay=0.1, patience=4, result_dir='results'):
    print "Start training..."

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    vargs_model = dict(num_blocks=num_blocks, dropout=dropout)

    shutil.copyfile(__file__, os.path.join(result_dir,'ResNet_module.py'))

    prev_val_acc = 0.0
    patience_counter = 0

    #Options: MomentumSGD, AdaGrad, AdaDelta
    Optimizer = optimizers.MomentumSGD

    print "model vargs {}, optimizer {}, batchsize {}".format(vargs_model, Optimizer, batchsize)

    np.random.seed(0)
    cp.random.seed(0)


    model = ResNet(**vargs_model)
    
    #Options: MomentumSGD, AdaGrad, AdaDelta
    optimizer = Optimizer()
    optimizer.setup(model)

    if weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    try:
        optimizer.lr = lr
    except:
        print 'cannot adjust lr for this optimizer'

    if gpu >= 0:
        model.to_gpu()
    
    N_train = train_data.shape[0]
    N_valid = val_data.shape[0]

    for epoch in range(1, epochs + 1):
        sum_loss, sum_accuracy = do_epoch(
            train_data, train_labels, model, optimizer, batchsize, train=True, finetune=False
        )
        msg = '\nepoch:{:02d}\ttrain mean loss={}, error={}'.format(
            epoch, sum_loss / N_train, 1. - sum_accuracy / N_train)
        print msg

        if epoch % valid_freq == 0:
            model.start_finetuning()

            print 'Start finetuning'
            sum_loss, sum_accuracy = do_epoch(
                train_data, train_labels, model, optimizer, batchsize, train=False, finetune=True
            )
            msg = '\nepoch:{:02d}\tfinetune mean loss={}, error={}'.format(
                epoch, sum_loss / N_train, 1. - sum_accuracy / N_train)
            print msg

            sum_loss, sum_accuracy = do_epoch(
                val_data, val_labels, model, optimizer, batchsize, train=False, finetune=False
            )
            msg = '\nepoch:{:02d}\ttest mean loss={}, error={}'.format(
                epoch, sum_loss / N_valid, 1. - sum_accuracy / N_valid)
            print msg


            if sum_accuracy <= prev_val_acc:
                patience_counter += 1
                if patience_counter > patience:
                    print '\nStopping early...'
                    break
            else:
                print 'Saving model...\n'
                serializers.save_hdf5( os.path.join(result_dir, 'epoch{:02d}_acc{:.3f}.model'.format(epoch, float(sum_accuracy) / N_valid)), 
                                        model)
                serializers.save_hdf5( os.path.join(result_dir, 'best.model'.format(epoch, float(sum_accuracy) / N_valid)), 
                                        model)

                prev_val_acc = sum_accuracy
                patience_counter = 0


        if epoch in lr_schedule:
            try:
                optimizer.lr *= lr_decay
            except:
                print 'cannot adjust lr for this optimizer'
            
            msg = '\nepoch:{:02d}\tAdjusting learning rate: {}'.format(
                epoch, optimizer.lr)
            print msg



def predict(modelfn, model_vargs, data, batchsize=128, gpu=0):
    assert gpu >= 0, "CPU support not yet implemented"

    model = ResNet(**model_vargs) 
    serializers.load_hdf5(modelfn, model)
    model.to_gpu()

    N = data.shape[0]
    prediction = np.zeros(N, dtype='int')   

    for i in range(0, N, batchsize):
        x_batch = data[i:i+batchsize]

        x_var = Variable(cuda.to_gpu(x_batch))

        prediction[i:i+batchsize] = cuda.to_cpu(model.predict(x_var))

    if N % batchsize != 0:
        x_batch = data[N - N % batchsize:]

        x_var = Variable(cuda.to_gpu(x_batch))

        prediction[N - N % batchsize:] = cuda.to_cpu(model.predict(x_var))

    return prediction

def predict_from_file(fn, data, model_vargs):
    module = imp.load_source('module', os.path.join(os.path.dirname(fn), 'ResNet_module.py'))
    return module.predict(fn, model_vargs, data, batchsize=128)
