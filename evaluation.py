import acclib
import numpy as np
import os
import itertools
import make_prediction_file as mpf

submissionsdir = 'all_submissions_val'

VOTING = 'normal'
#VOTING = 'weighted'
#VOTING = 'class_weighted'
MAX_MODELS_REMOVED = 0

def find_the_best():
    #This def attempts to find the best combination of models for the bagging approach and find the best voting scheme:
    #1. majority vote
    #2. weighted majority voting: the weight of the vote is determined by the overal accuracy of that respective model 
    
    #first we do this only for the total model, using all available submissions
    #I did this because i needed the predictions,labels and prediction_files variables
    predictions, votes, labels, prediction_files = mpf.check_vote_accuracy(submissionsdir)
    print '-------------------------------'
    
    #first i check the accuracy for the full model   
    accuracy = acclib.get_accuracy_per_class(votes, labels)
    prediction_files = list(prediction_files)

    bestAccuracy = 0.0
    bestFiles = []
    bestCombination = []
    total = 0
    
    #get the model_acc
    print 'GETTING THE MODEL ACCURACY'
    model_acc = np.zeros(len(prediction_files))
    model_acc_per_class = np.zeros((len(prediction_files), len(np.unique(labels))))
    for i in range(len(prediction_files)):
        print i, prediction_files[i]
        votes = mpf.vote_prediction(predictions[:, i])
        model_acc[i], model_acc_per_class[i, :] = acclib.get_accuracy_per_class(votes, labels)
    print model_acc
    print 'Accuracy per model per class', model_acc_per_class
    print 'Mean accuracy per class', np.mean(model_acc_per_class[:,0]), np.mean(model_acc_per_class[:,1]), np.mean(model_acc_per_class[:,2]), np.mean(model_acc_per_class[:,3])
    
    for m in range(model_acc_per_class.shape[1]):
        #model_acc_per_class[:,m] = model_acc_per_class[:,m] - np.min(model_acc_per_class[:,m])
        model_acc_per_class[:,m] = model_acc_per_class[:,m] / np.max(model_acc_per_class[:,m])  
    
    
    print model_acc_per_class    
    import time
    start = time.time()
    #then we try all possible combinations
    for i in range(predictions.shape[1], 0, -1)[:MAX_MODELS_REMOVED+1]:
        combinations = itertools.combinations(range(predictions.shape[1]), i)
        for k, c in enumerate(combinations):
            combination_files = [prediction_files[x] for x in c]
            print 'Predicting using: {}'.format(combination_files)
            if VOTING == 'normal':
                votes = mpf.vote_prediction(predictions[:, list(c)])
            if VOTING == 'weighted':
                votes = mpf.vote_prediction_weighted(predictions[:, list(c)], model_acc[list(c)])
            if VOTING == 'class_weighted':
                votes = mpf.vote_prediction_class_weighted(predictions[:, list(c)], model_acc_per_class[list(c), :])
                #print 'Voting performance weighted is {:.1f}%'.format(100. * np.mean(votes_weighted == labels))
            
            accuracy, acc_per_class = acclib.get_accuracy_per_class(votes, labels)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestCombination = c
                bestFiles = combination_files
            print 'Voting performance is {:.1f}%'.format(100. * np.mean(votes == labels))
            total += 1
        print i, total
          
    print 'done in:' , time.time() - start, 'seconds'
    print 'total:', total      
    print 'BEST MODEL'
    print 'voting:', VOTING
    print 'total amount of models used:', len(bestCombination)
    print bestFiles
    print bestAccuracy
    print bestCombination
    #mpf.make_vote_prediction_file(title='vote_class_weighted', submissions_folder='submissions', model_acc_per_class=model_acc_per_class)
    return votes

find_the_best()   
  


