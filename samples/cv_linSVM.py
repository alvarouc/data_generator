#!/usr/bin/python
'''
Sample code for cross validation using the data generator
'''
from smri_gen import DataGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc
from sklearn.cross_validation import train_test_split
#import progressbar as pb
#import multiprocessing as mp
#from functools import partial

N_SAMPLES = 5000


def fit_test(clf, train_tuple, test_tuple):
    '''
    fit_test function that fits a classifier in train_tuple and
    report AUC results on test_tuple
    The tuples should be given as (data, label)
    '''
    data_train, labels_train = train_tuple
    data_test, labels_test = test_tuple
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    clf.fit(data_train, labels_train)
    fpr, tpr, _ = rc(labels_test, clf.predict_proba(data_test)[:, 1])
    return auc(fpr, tpr)


def classify(data, labels, seed):
    '''
    Classifier function that receives data, labels and a seed for
    random number generation.
    '''
    # Data split in 90% train 10% test
    data_train, data_test, labels_train, labels_test\
        = train_test_split(data, labels, test_size=.1, random_state=seed)

    new_data = DataGenerator(data_train,
                             labels_train, 15).generate(N_SAMPLES)
    new_labels = [0]*N_SAMPLES + [1]*N_SAMPLES

    clf = SVC(kernel='linear', C=0.05, probability=True)
    # Fit on simulated data
    result_sim = fit_test(clf, (new_data, new_labels),
                          (data_test, labels_test))

    # Fit on real data
    result = fit_test(clf, (data_train, labels_train),
                      (data_test, labels_test))

    print 'Process %d: (%.2f vs %.2f) AUC' % (seed, result_sim, result)
    return(result_sim, result)


def main():
    '''
    Demo of cross validation using Linear SVM
    '''
    #path = '/export/mialab/users/alvaro/data/mega/'
    path = '/home/aulloa/data/mega/'
    data = np.load(path + 'dataQC.npy')
    labels = np.load(path + 'labelQC.npy').ravel()

    iterations = 4

    #p = mp.Pool(processes= 4)
    #fun = partial(classify, data=data, labels=labels, seed)
    #results = p.map(fun,range(iterations))
    #results = map(fun,range(iterations))
    results = [classify(data, labels, seed) for seed in range(iterations)]
    np.save('results.npy', np.array(results))

    print results


if __name__ == '__main__':
    main()

