"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rt
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import BagLearner as bl
from matplotlib.ticker import MaxNLocator

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    # if using files other than Istanbul.csv, the use the following code to read file
    # data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    data = np.array(pd.read_csv(inf,sep=",").ix[:,1:])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    learner = rt.RTLearner(leaf_size=50, verbose = True) # create a random forest learner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]


    repeat = 10
    leafsize = 10
    rmse_in = np.zeros([leafsize,repeat])
    c_in = np.zeros([leafsize,repeat])
    rmse_out = np.zeros([leafsize,repeat])
    c_out = np.zeros([leafsize,repeat])

    for i in range(leafsize):
        for j in range(repeat):

            data_new = data
            np.random.shuffle(data_new)
            # separate out training and testing data
            trainX = data_new[:train_rows, 0:-1]
            trainY = data_new[:train_rows, -1]
            testX = data_new[train_rows:, 0:-1]
            testY = data_new[train_rows:, -1]

            learner = rt.RTLearner(leaf_size=i+1, verbose=False)  # create a random forest learner
            # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=i+1, boost=False, verbose=False)
            learner.addEvidence(trainX, trainY)  # train it

            # evaluate in sample
            predY_in = learner.query(trainX)  # get the predictions
            rmse_in[i,j] = math.sqrt(((trainY - predY_in) ** 2).sum() / trainY.shape[0])
            # c_in[i,j] = np.corrcoef(predY_in, y=trainY)[0,1]

            # evaluate out of sample
            predY_out = learner.query(testX)  # get the predictions
            rmse_out[i,j] = math.sqrt(((testY - predY_out) ** 2).sum() / testY.shape[0])
            # c_out[i,j] = np.corrcoef(predY_out, y=testY)[0,1]

    rmse_in_average = np.nanmean(rmse_in, axis=1)
    rmse_out_average = np.nanmean(rmse_out, axis=1)
    rmse = np.stack((rmse_in_average,rmse_out_average),axis=-1)
    print "RMSE", rmse_in
    print "RMSE_average", rmse

    df = pd.DataFrame(rmse)
    df.columns = ['In sample RMSE','Out of sample RMSE']
    ax = df.plot(title = 'In and out of sample RMSE vs. bag size with 200 leaf')
    ax.set_xlabel('bag size')
    ax.set_ylabel('RMSE')
    ax.set_ylim([0.0, 0.010])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
