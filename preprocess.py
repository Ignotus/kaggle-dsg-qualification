import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def get_preprocessed_dir(datadir, shape, normalize_with_std, reflectance): 
    folder_name = 'preprocessed_%d_%d' % shape

    if not normalize_with_std:
        folder_name += '_nostd'

    if reflectance:
        folder_name += '_refl'

    return os.path.join(datadir, folder_name)

def get_roof_data(datadir='data', 
                  shape=(32, 32), 
                  augmented=True, 
                  normalize_with_std=True,
                  reflectance=False):
                      
    if not os.path.exists('data/'):
        print "Missing 'data' folder, please create one in the same directory as the 'main.py' file"
        quit(1)

    if not (os.path.exists('data/id_train.csv') and os.path.exists('data/roof_images/')):
        print "Missing training data, make sure to place the id_train.csv file and roof_images folder inside the data folder"
        quit(1)
    
    result_path = get_preprocessed_dir(datadir, shape, normalize_with_std, reflectance)
    if not os.path.exists(result_path):
        preprocess(datadir, shape, normalize_with_std, reflectance)

    archive = np.load(os.path.join(result_path, 'train.npz'))
    train_data = archive['data']
    train_labels = archive['labels']

    archive = np.load(os.path.join(result_path, 'valid.npz'))
    valid_data = archive['data']
    valid_labels = archive['labels']
    
    test_data = None
    test_ids = None
    
    
    archive = np.load(os.path.join(result_path, 'test.npz'))
    test_data = archive['data']
    test_ids = archive['ids']
        

    if augmented:
        train_data, train_labels = generate_augmented_data(train_data, train_labels)

    return train_data, train_labels, valid_data, valid_labels, test_data, test_ids


def preprocess(datadir, shape, normalize_with_std=False, reflectance=False):
    print "Preprocessing..."

    print "\tReading images from file..."
    if reflectance:
        path = os.path.join(datadir, 'intrinsic_images/')
    else:
        path = os.path.join(datadir, 'roof_images/')

    fn_images = os.listdir(path)

    data_ids = np.hstack((get_train_info(datadir)[:, 0], get_test_ids(datadir))).tolist()
    data_ids = set(map(int, data_ids))

    ids = []
    images = []

    if not reflectance:
        for fn in fn_images:
            if len(fn) > 4 and fn[-4:] == '.jpg':
                idx = int(fn[:-4])
                # Loads only images that are presented in the training or testing set to
                # reduce a memory consumption
                if idx in data_ids:
                    ids.append(idx)
                    images.append(ndimage.imread(os.path.join(path, fn)))
    else:
        # Reading reflectance images generated by intrinsic image decomposition
        for fn in fn_images:
            if len(fn) > 6 and fn[-6:] == '-r.png':
                idx = int(fn[:-6])
                if idx in data_ids:
                    ids.append(idx)
                    images.append(ndimage.imread(os.path.join(path, fn)))

    ids = np.array(ids, dtype='int')
    images = np.asarray(images)

    images = resize(images, shape)
    
    images = switch_channels(np.array(images))

    # images = contrast_normalize(images)

    all_train_data, all_train_labels, test_data, test_ids = collect_data(datadir, images, ids)

    train_data = all_train_data[:-1000]
    train_labels = all_train_labels[:-1000]
    valid_data = all_train_data[-1000:]
    valid_labels = all_train_labels[-1000:]


    print "\tNormalizing..."
    mean = np.mean(train_data, axis=(0, 2, 3))
    
    for i in range(3):
        train_data[:, i, :, :] -= mean[i]
        valid_data[:, i, :, :] -= mean[i]
        test_data[:, i, :, :] -= mean[i]
    if normalize_with_std:
        std = np.std(train_data, axis=(0, 2, 3))
        std += 1e-5
    
    
    if normalize_with_std:
        for i in range(3):
            train_data[:, i, :, :] /= std[i]
            valid_data[:, i, :, :] /= std[i]
            test_data[:, i, :, :] /= std[i]

    train_data = train_data.astype('float32')
    train_labels = train_labels.astype('int32')
    valid_data = valid_data.astype('float32')
    valid_labels = valid_labels.astype('int32')
    test_data = test_data.astype('float32')

    result_path = get_preprocessed_dir(datadir, shape, normalize_with_std, reflectance)
    os.mkdir(result_path)

    np.savez(os.path.join(result_path, 'train.npz'), data=train_data, labels=train_labels)
    np.savez(os.path.join(result_path, 'valid.npz'), data=valid_data, labels=valid_labels)
    np.savez(os.path.join(result_path, 'test.npz'), data=test_data, ids=test_ids)

    return result_path

def collect_data(datadir, images, ids):
    print "\tCollecting data..."
    mapping = dict(zip(ids, np.arange(ids.shape[0])))
    
    train_info = get_train_info(datadir)

    train_data = images[[mapping[x] for x in train_info[:, 0]]]
    images[[mapping[x] for x in train_info[:, 0]]] = 0 
    train_labels = train_info[:, 1]

    test_ids = get_test_ids(datadir)
    test_data = images[[mapping[x] for x in test_ids]]
    
    images = None
    return train_data, train_labels, test_data, test_ids


def get_train_info(datadir):
    return np.loadtxt(os.path.join(datadir, 'id_train.csv'), delimiter=',', skiprows=1, dtype='int')

def get_test_ids(datadir):
    return np.loadtxt(os.path.join(datadir, 'sample_submission4.csv'), delimiter=',', skiprows=1, dtype='int')[:, 0]

def resize(images, shape):
    print "\tResizing images..."
    result = np.zeros((len(images),) + shape + (3,))

    for i in range(len(images)):
        result[i, ...] = misc.imresize(images[i], shape, interp='bicubic')

    return result

def switch_channels(images):
    return images.transpose(0, 3, 1, 2)

    
def contrast_normalize(images, epsilon=1e-2):
    print "\tNormalizing..."
    # Per image normalization
    mean = np.mean(images, axis=(2, 3))
    sigma = np.std(images, axis=(2, 3))

    sigma[sigma < epsilon] = 1.0
    return ((images - mean[..., None, None]) / sigma[..., None, None])

def generate_augmented_data(data, labels):
    print "Generating augmented data..."
    N = data.shape[0]
    data_augmented = np.empty((N * 8,) + data.shape[1:]).astype('float32')
    labels_augmented = np.empty(N * 8).astype('int32')

    for i in range(data.shape[0]):
        data_augmented[i] = data[i]
        data_augmented[1 * N+i] = data[i, ..., ::-1, :]
        data_augmented[2 * N+i] = data[i, ..., :, ::-1]
        data_augmented[3 * N+i] = data[i, ..., ::-1, ::-1]

        data_augmented[4 * N+i] = data_augmented[i].transpose(0, 2, 1)
        data_augmented[5 * N+i] = data_augmented[1 * N+i].transpose(0, 2, 1)
        data_augmented[6 * N+i] = data_augmented[2 * N+i].transpose(0, 2, 1)
        data_augmented[7 * N+i] = data_augmented[3 * N+i].transpose(0, 2, 1)

        labels_augmented[i] = labels[i]
        labels_augmented[1 * N+i] = labels[i]
        labels_augmented[2 * N+i] = labels[i]
        labels_augmented[3 * N+i] = labels[i]

        flipped_label = labels[i]
        if flipped_label == 1:
            flipped_label = 2
        elif flipped_label == 2:
            flipped_label = 1

        labels_augmented[4 * N+i] = flipped_label
        labels_augmented[5 * N+i] = flipped_label
        labels_augmented[6 * N+i] = flipped_label
        labels_augmented[7 * N+i] = flipped_label


    return data_augmented, labels_augmented

if __name__ == '__main__':
    train_data, train_labels, test_data, test_ids,_,_ = get_roof_data(shape=(32,32))
    
    print train_data[:2], train_labels[:2]
