from utils import generate_results_csv
from utils import create_directory
from utils import read_dataset
from utils import transform_mts_to_ucr_format
from utils import visualize_filter
from utils import viz_for_survey_paper
from utils import viz_cam

import os
import numpy as np
import sys
import sklearn

import utils
import constants
from constants import CLASSIFIERS
from constants import ARCHIVE_NAMES
from constants import ITERATIONS
from utils import read_all_datasets


def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    ######################################
    #x_train=poscode(x_train)
    #x_test=poscode(x_test)
    ##########################################

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))#类数

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))#将标签转为可计算的
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()#转化为一列
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)#每一个类对应一个编码，值为一的位置就是类别标签

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))#序列个数*序列长度*1
        #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        x_train = x_train.reshape((x_train.shape[0], 47, 10))# \u5e8f\u5217\u4e2a\u6570*time_step23*feature_10
        x_test = x_test.reshape((x_test.shape[0], 47, 10))

    input_shape = x_train.shape[1:]#序列长度*1
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'selfattention':
        import selfattention3
        return selfattention3.Classifier_SELF(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet_causal':
        import resnet_causal
        return resnet_causal.Classifier_RESNET_CAUSAL(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet_convattention_causal':
        import resnet_convattention_causal
        return resnet_convattention_causal.Classifier_RESNET_CONVATTENTION_CAUSAL(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet_blockattention_causal':
        import resnet_blockattention_causal
        return resnet_blockattention_causal.Classifier_RESNET_BLOCKATTENTION_CAUSAL(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet_in_out':
        import resnet_in_out
        return resnet_in_out.Classifier_RESNET_IN_OUT(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'copy':
        import resnet_copy
        return resnet_copy.Classifier_RESNET_COPY(output_directory, input_shape, nb_classes, verbose)
############################################### main

# change this directory for your machine
#root_dir = '/b/home/uha/hfawaz-datas/dl-tsc-temp/'
root_dir = 'E:\文献\Self-Attention-Deep-Learing-test\ljw'

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + os.sep +'results'+os.sep + classifier_name + os.sep + archive_name + trr + os.sep

                for dataset_name in constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + os.sep
                    
                    if os.path.exists(output_directory+'df_metrics.csv'):
                        print('\t\t\t\tdataset_name: '+classifier_name+'的'+dataset_name+'\t'+'Already done')
                        continue
                    
                    else:
                        
                        create_directory(output_directory)

                        fit_classifier()

                        print('\t\t\t\tDONE')

                    # the creation of this directory means
                        create_directory(output_directory + os.sep+'DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':#
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1] == 'viz_cam':#
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + os.sep+'results'+os.sep + classifier_name + os.sep + archive_name + itr + os.sep + \
                       dataset_name + os.sep

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

        fit_classifier()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + os.sep+'DONE')
