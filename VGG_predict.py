# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:27:15 2016

@author: Riaan
"""

from models.VGG_16 import VGG_16_test, VGG_19_test
from keras.optimizers import SGD, Adagrad
import numpy as np 
import preprocess as p

from make_prediction_file import make_prediction_file

LR = 0.0001
# weightspath is none, for now!
def build_model(optimizer, model, shape):
    model = model(shape = shape) 
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def make_predictions(shape, model):
    
    train_data, train_ids, valid_data, valid_labels, test_data, test_ids = p.get_roof_data(shape=(shape,shape))
    
    print '\tInitializing model'
    opt = Adagrad(lr = LR)
    model = build_model(opt, model, shape)
    
    print '\tCreating predictions'
    pred = model.predict_classes(test_data, 
                          batch_size = 20, 
                          verbose=0)
    
    pred_valid = model.predict_classes(valid_data, 
                          batch_size = 20, 
                          verbose=0)
    
    pred = np.array([x + 1 for x in list(pred)])
    pred_valid = np.array([x + 1 for x in list(pred_valid)])
    print '\tWriting to file'
    make_prediction_file(test_ids, pred,'vgg_predictions', 
                         valid_labels= valid_labels,
                         valid_predictions= pred_valid)
                         
    
if __name__ == '__main__':
    make_predictions()
    