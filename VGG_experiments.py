# -*- coding: utf-8 -*-
"""
Creating the VGG-16, training and validation

Author Riaan Zoetmulder
"""

from models.VGG_16 import VGG_16, VGG_19
from keras.optimizers import SGD, Adagrad
import numpy as np 
import preprocess as p
from keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
import shutil



# No prior weights
WeightsPath = None
ITERATIONS = 1000
BATCH = 30


# function to store the best model
def store(model, final_accuracy, modelname):
    
    # if no directory exists create one
    path = 'saved_models/' + 'best_model_'  + modelname
    if not os.path.exists(path):
        os.makedirs(path)
        
    
    # if one exists remove it!
    else:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    
    nm = '16'
    if modelname == 'VGG_19':
        nm = '19'
    
    
    fl = path + '/weights_' + nm+ '.h5'
    model.save_weights(fl)
    

# weightspath is none, for now!
def build_model( optimizer, model,  shape, WeightsPath = None, 
                dropout = 0.5):
                    
    model = model(WeightsPath, DO = dropout, shape= shape) 
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
def train(dir_name, modelname, weights,
          model,shape, train_data, train_labels,
          valid_data, valid_labels, rand = True):
    
    PATIENCE = 2
    BATCH = 29
    LR = 0.0005
    DO = 0.654
        
    if rand:
        PATIENCE = random.randint(1, 4)
        BATCH = random.randint(25, 80)
        LR = random.uniform(0.0007, 0.0001)
        DO = random.uniform(0.62, 0.68)
        
    # build the model
    # opt = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adagrad(lr = LR)
    model = build_model(opt, model, shape, WeightsPath = weights, dropout = DO)
    
    
    # training targets
    t_trgt = np.zeros((train_labels.shape[0], 4))
    t_trgt[np.arange(train_labels.shape[0]), train_labels] = 1
    
    # validation targets
    v_trgt = np.zeros((valid_labels.shape[0], 4))
    v_trgt[np.arange(valid_labels.shape[0]), valid_labels] = 1
    

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience= PATIENCE)
    
    if not os.path.exists('saved_models/'+ modelname + '/' ):
        os.makedirs('saved_models/'+ modelname +'/')
    if not os.path.exists('saved_models/'+ modelname + '/'+ dir_name):
        os.makedirs('saved_models/'+ modelname +'/' + dir_name)
        
  
    print '\tTraining Model'     
    model.fit(train_data, t_trgt,
                      validation_data= (valid_data, v_trgt),
                      callbacks=[early_stopping],
                      nb_epoch=ITERATIONS,
                      batch_size=BATCH, 
                      verbose = 1)
    
    
    
    loss_and_metrics = model.evaluate(valid_data, v_trgt, batch_size=20)
    
    print 'Neural Network Cross entropy: ' + str(loss_and_metrics[0]) + ' Accuracy: '+ str(loss_and_metrics[1]) 
    
    # write information to file
    file = open('saved_models/'+ modelname + '/' + dir_name + "/settings.txt", "w")
    
    file.write('Learning rate: '+ str(LR)+ '\n')
    file.write('Batch size: '+ str(BATCH)+ '\n')
    file.write('Patience: ' + str(PATIENCE)+ '\n')
    file.write('Dropout: ' + str(DO)+ '\n')
    file.write('Accuracy: ' + str(loss_and_metrics[1]) + '\n')
    
    file.close()
    
    return str(loss_and_metrics[1]), model


# loop to run experiments and determine highest values
def random_hyperparameter_search(modelname, model, weights, shape, train_data,
                                 train_labels, valid_data, valid_labels):
    accuracy = 0.0
    exp_folder = None
    for x in range(0,25):
        
        dirname = 'Exp_' + str(x) 
        acc, mdl = train(dirname, modelname, weights, model, shape, train_data, train_labels, valid_data, valid_labels)
        
        if acc > accuracy:
            # store all data pertaining to accuracy + model
            accuracy = acc
            exp_folder = dirname
            print 'New highest accuracy: ' + str(acc)
            
            # store model
            store(mdl, accuracy, modelname)
    
    print 'highest found FINAL accuracy for VGG: ' + str(accuracy)
    print 'Parameters can be found in: ' + exp_folder
        
    
    
    

