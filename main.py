import argparse
import preprocess
import make_prediction_file
import sys
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='Chooses a model', default='GoogleNet')
    parser.add_argument('--epoch', help='Number of epochs', default=30, type=int)
    parser.add_argument('--weights', help='Specifies a weight file', default='model.npz')
    parser.add_argument('--train', help='Enables the training mode, disabled by default', default=False, action='store_true')
    parser.add_argument('--tune', help='Enables parameters tuning mode, disabled by default', default=False, action='store_true')
    parser.add_argument('--name', help='Specifies a model name', default='googlenet')
    parser.add_argument('--img_dim', help='Specifies an image dimension', default=64, type=int)
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()

    if args.model == 'GoogleNet':
        train_data, train_labels, valid_data, valid_labels, test_data, test_ids = preprocess.get_roof_data(shape=(args.img_dim, args.img_dim),
                                                                                                           augmented=args.train,
                                                                                                           normalize_with_std=(args.version != 'v1'),
                                                                                                           reflectance=False)
        train_labels -= 1
        valid_labels -= 1

        if args.train:
            from models.googlenet import train

            if not args.tune:
                acc = train(train_data, train_labels, valid_data, valid_labels,
                            args.version,
                            0.0051616, 0.3625, 37, 0.00001, args.epoch)
            else:
                from models.googlenet import random_search_hyperparameters
                random_search_hyperparameters(train_data, train_labels, valid_data, valid_labels, args.version)
        else:
            from models.googlenet import predict

            prediction, validation_prediction = predict(valid_data, valid_labels, test_data, model=args.version, model_path=args.weights)

            valid_labels += 1
            make_prediction_file.make_prediction_file(test_ids, prediction, args.name, valid_labels, validation_prediction)
    
    # If VGG model is selected
    elif args.model =='VGG':
        train_data, train_labels, valid_data, valid_labels, test_data, test_ids = preprocess.get_roof_data(shape=(args.img_dim, args.img_dim),
                                                                                                           augmented=args.train,
                                                                                                           normalize_with_std=True,
                                                                                                           reflectance=False)
        # reduce the labels by 1
        train_labels = np.array([x - 1 for x in train_labels ])
        valid_labels =np.array([x - 1 for x in valid_labels ])
                                                                                     
        if args.train:
            
            from models.VGG_16 import VGG_16, VGG_19
            
            # for experiments with the VGG_16
            
            if args.version == '16':
                
                if not args.tune:
                    
                    # Train the model with optimized hyperparameters
                    from VGG_experiments import  train
                    
                    train('finetuned_16', 'VGG_16', 'vgg16_weights.h5',
                          VGG_16 , args.img_dim, train_data, train_labels,
                          valid_data, valid_labels, rand = False)
                else:
                    
                    from VGG_experiments import random_hyperparameter_search
                     
                    random_hyperparameter_search('VGG_'+ args.version, VGG_16, 'vgg16_weights.h5', args.img_dim, train_data,
                                 train_labels, valid_data, valid_labels)
                                     
            # for experiments with the VGG_19
            elif args.version == '19':
                
                if not args.tune:
                    
                    # Train the model with optimized hyperparameters
                    train('finetuned_19', 'VGG_19', 'vgg19_weights.h5',
                          VGG_19 , args.img_dim, train_data, train_labels,
                          valid_data, valid_labels, rand = False)
                
                else:
                    
                    # do a random hyperparameter search
                    from VGG_experiments import random_hyperparameter_search
                     
                    random_hyperparameter_search('VGG_'+ args.version, VGG_19, 'vgg19_weights.h5', args.img_dim, train_data,
                                 train_labels, valid_data, valid_labels)
                                     
            # if version not recognized.
            else: 
                sys.exit('VGG version not recognized. Please enter correctly.')
                
        else:
            from VGG_predict import make_predictions
            from models.VGG_16 import VGG_16_test, VGG_19_test
            
            if args.version == '16':
                make_predictions(args.img_dim, VGG_16_test)
            elif args.version =='19':
                make_predictions(args.img_dim, VGG_19_test)
            else:
                sys.exit('cannot find model you have specified')
    
    elif args.model == 'ResNet':
        import models.ResNet as ResNet

        train_data, train_labels, valid_data, valid_labels, test_data, test_ids = preprocess.get_roof_data(augmented=True, shape=(64, 64))

        if args.train:

            ResNet.train(train_data, train_labels, valid_data, valid_labels, dropout=0.62, num_blocks=3, lr=0.007, weight_decay=0.004)

        else:
            model_vargs = dict(dropout=0.62, num_blocks=3)
            fn = 'results/best.model'
            valid_predictions = ResNet.predict(fn, model_vargs, valid_data)
            test_predictions = ResNet.predict(fn, model_vargs, test_data)
            make_prediction_file.make_prediction_file(test_ids, test_predictions, 'Resnet805_64_64', valid_labels=valid_labels, valid_predictions=valid_predictions)

                                                                                                          
                                                                                                          
        
