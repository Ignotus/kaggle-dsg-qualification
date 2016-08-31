import numpy as np
import os

def make_prediction_file(test_ids, test_predictions, title, valid_labels=None, valid_predictions=None):
    assert test_ids.shape == test_predictions.shape, "Shape of test ids is not equal to shape of predictions"

    if not os.path.exists('submissions/'):
        os.mkdir('submissions')

    array = np.array([ test_ids, test_predictions ], dtype='int').T
    np.savetxt('submissions/submission_%s.csv' % title, array, fmt='%i', comments='', delimiter=",", header='Id,label')
    print 'Test file created.'

    if isinstance(valid_predictions, np.ndarray) and isinstance(valid_labels, np.ndarray):
        assert valid_labels.shape == valid_predictions.shape, "Shape of valid labels is not equal to shape of predictions"

        if not os.path.exists('submissions_val/'):
            os.mkdir('submissions_val')

        array = np.array([ valid_labels, valid_predictions ], dtype='int').T
        np.savetxt('submissions_val/submission_%s_val.csv' % title, array, fmt='%i', comments='', delimiter=",", header='Label,Prediction')
        print 'Validation file created with accuracy of {:.1f}%'.format(100. * np.mean(valid_labels == valid_predictions))
    else:
        print 'Warning: No validation prediction has been created.'

def vote_prediction(predictions):
    # Compute votes by taking maximum voted.
    if len(predictions.shape) == 1:
        predictions = np.reshape(predictions, (predictions.shape[0], 1))
        #print 'reshaped to Nx1 instead of N'
    N = len(predictions)
    number_of_agreements = 0.
    impasse = 0.
    votes = np.zeros(N, dtype='int')
    for i in range(N):
        bincount = np.bincount(np.array([predictions[i]]).flatten())

        assert bincount[0] == 0, 'A prediction file contains a 0 prediction, not a valid class.'
        if predictions.shape[1] == bincount.max():
            number_of_agreements += 1
        if np.sum(bincount == bincount.max()) > 1:
            impasse += 1

        votes[i] = np.argmax(bincount)

    print 'Total number of agreements: {:.2f}% \nTotal number of disagreements: {:.2f}%'.format(100. * number_of_agreements / N, 100. - 100. * number_of_agreements / N)
    print 'Total number of impasses: {:.2f}%'.format(100. * impasse / N)
    return votes
    
#def vote_prediction_weighted(predictions, model_acc = [0.809, 0.796, 0.8, 0.807, 0.806, 0.805, 0.813, 0.832, 0.814, 0.794, 0.805, 0.816, 0.826, 0.828, 0.806, 0.82, 0.82, 0.818]): #, model_acc_per_class):
def vote_prediction_weighted(predictions, model_acc = [0.809, 0.813,0.8, 0.807, 0.809, 0.808, 0.817, 0.806, 0.813, 0.812, 0.832, 0.814, 0.794, 0.805, 0.816, 0.825, 0.826, 0.806, 0.82, 0.818]): #, model_acc_per_class):
    # Compute votes by adding the accuracy prior to the model vote.
    #print 'Computing votes by multiplying the vote weight by the respective model accuracy'
    N = len(predictions)
    number_of_agreements = 0.
    impasse = 0.

    votes = np.zeros(N, dtype='int')
    for i in range(N):
        bincount = weighted_bincount(predictions[i], model_acc)

        assert bincount[0] == 0, 'A prediction file contains a 0 prediction, not a valid class.'
        if len(np.nonzero(bincount)[0]) == 1:
            number_of_agreements += 1
        if np.sum(bincount == bincount.max()) > 1:
            impasse += 1

        votes[i] = np.argmax(bincount)

    print 'Total number of agreements: {:.2f}% \nTotal number of disagreements: {:.2f}%'.format(100. * number_of_agreements / N, 100. - 100. * number_of_agreements / N)
    print 'Total number of impasses: {:.2f}%'.format(100. * impasse / N)
    return votes

def vote_prediction_class_weighted(predictions, model_acc_per_class): #, model_acc_per_class):
    # Compute votes by adding the accuracy prior to the model vote.
    #print 'Computing votes by multiplying the vote weight by the respective model accuracy'
    N = len(predictions)
    number_of_agreements = 0.
    impasse = 0.

    votes = np.zeros(N, dtype='int')
    for i in range(N):
        bincount = class_weighted_bincount(predictions[i], model_acc_per_class)

        assert bincount[0] == 0, 'A prediction file contains a 0 prediction, not a valid class.'
        if len(np.nonzero(bincount)[0]) == 1:
            number_of_agreements += 1
        if np.sum(bincount == bincount.max()) > 1:
            impasse += 1

        votes[i] = np.argmax(bincount)

    print 'Total number of agreements: {:.2f}% \nTotal number of disagreements: {:.2f}%'.format(100. * number_of_agreements / N, 100. - 100. * number_of_agreements / N)
    print 'Total number of impasses: {:.2f}%'.format(100. * impasse / N)
    return votes

def weighted_bincount(prediction, model_acc):
    assert len(prediction) == len(model_acc), "prediction and model_acc are not of equal length" + len(prediction) + len(model_acc)
    bincount = np.zeros(5)
    for i, p in enumerate(prediction):
        bincount[p] += model_acc[i]
    return bincount
 
def class_weighted_bincount(prediction, model_acc_per_class):
    bincount = np.zeros(5)
    for i, p in enumerate(prediction):
        bincount[p] += model_acc_per_class[i][p-1]
    return bincount
               

def get_prediction_paths(folder):
    # Get all prediction files.
    listdir = os.listdir(folder)
    prediction_files = []
    for fn in listdir:
        if len(fn) > 4 and fn[-4:] == '.csv':
            # ignore votes
            if 'vote' in fn:
                continue

            # ignore commented files
            if fn[0] == '#':
                continue

            prediction_files.append(fn)

    return prediction_files

def make_vote_prediction_file(title='vote', submissions_folder='submissions', model_acc_per_class=None):
    
    prediction_files = get_prediction_paths(submissions_folder)

    # Load test ids from first prediction file.
    test_ids = np.loadtxt(os.path.join(submissions_folder, prediction_files[0]), dtype='int', delimiter=',', skiprows=1)[:,0]

    N = len(test_ids)

    predictions = np.zeros((N, len(prediction_files)), dtype='int')

    # Read the prediction files
    for i in range(predictions.shape[1]):
        predictions[:, i] = np.loadtxt(os.path.join(submissions_folder, prediction_files[i]), dtype='int', delimiter=',', skiprows=1)[:,1]
    
    if model_acc_per_class == None:
        print 'making test file with normal voting'
        votes = vote_prediction(predictions)
    else:
        print 'making test file with class weighted voting'
        votes = vote_prediction_class_weighted(predictions, model_acc_per_class)

    make_prediction_file(test_ids, votes, title)

def check_vote_accuracy(submissions_folder='submissions_val'):
    prediction_files = get_prediction_paths(submissions_folder)

    print 'Predicting using: {}'.format(prediction_files)

    # Load test ids from first prediction file.
    labels = np.loadtxt(os.path.join(submissions_folder, prediction_files[0]), dtype='int', delimiter=',', skiprows=1)[:,0]

    N = len(labels)

    predictions = np.zeros((N, len(prediction_files)), dtype='int')

    # Read the prediction files
    for i in range(predictions.shape[1]):
        predictions[:, i] = np.loadtxt(os.path.join(submissions_folder, prediction_files[i]), dtype='int', delimiter=',', skiprows=1)[:,1]
    
    votes = vote_prediction(predictions)

    assert labels.shape == votes.shape, 'Labels and votes dont have the same shape.'
    print 'Voting performance is {:.1f}%'.format(100. * np.mean(votes == labels))
    
    return predictions, votes, labels, prediction_files

if __name__ == '__main__':
    check_vote_accuracy()
    make_vote_prediction_file()
    


