"""
Random Tree learner with just one tree
"""

import numpy as np
import random

class RTLearner(object):
    def __init__(self, leaf_size=3, verbose=False):
        self.leaf_size = leaf_size
        pass  # move along, these aren't the drones you're looking for

    def addEvidence(self, X_data, Y_data):
        # save the model to random_tree function for query purpose
        self.random_tree = self.BuildTree(X_data, Y_data)

    def BuildTree(self, X_data, Y_data):
        # if row <= leaf size, then return the average of all observations
        if X_data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(Y_data), -1, -1]])
        # if Y has only one unique value, then return that value without any split
        if len(np.unique(Y_data)) == 1:
            return np.array([[-1, Y_data[0], -1, -1]])

        # initialize the left and right tree
        left_tree = 0
        right_tree = 0
        
        # randomly select the feature
        feature = random.choice(range(X_data.shape[1]-1))
        
        # randomly select two values from that selected feature to calculate split value
        split_val = np.mean(random.sample(X_data[:,feature],2))
        left_tree = X_data[X_data[:, feature] <= split_val].shape[0]
        right_tree = X_data[X_data[:, feature] > split_val].shape[0]
        
        # recursive call of the BuildTree function to finish building the tree until it reaches termination conditions
        left_tree = self.BuildTree(X_data[X_data[:, feature] <= split_val], Y_data[X_data[:, feature] <= split_val])
        right_tree = self.BuildTree(X_data[X_data[:, feature] > split_val], Y_data[X_data[:, feature] > split_val])
        
        # storage node and random feature values in tree variable
        tree = [feature, split_val, 1, left_tree.shape[0] + 1]
        
        # return the stacked array of all nodes and random features during recursive calls
        return np.vstack((tree, left_tree, right_tree))

    # define query function to calculate the predicted Y values
    def query(self, X_test):
        # initialize an empty array for storaging predicted Y values
        predictY = np.empty(shape=(X_test.shape[0],))
        # iterate all values in Y array
        for i in range(0, len(X_test)):
            predictY[i] = self.query_point(X_test[i], 0)
        return predictY
    
    # define a recursive function call to calculate predicted Y value using trained model
    def query_point(self, point, index):
        
        node = self.random_tree[index]
        # if only one leaf
        if node.item(0) == -1: return node.item(1)
        # query on the left tree
        elif point[int(node.item(0))] <= node.item(1): return self.query_point(point, int(node.item(2)) + index)
        # query on the right tree
        else: return self.query_point(point, int(node.item(3)) + index)
        

'''
if __name__ == "__main__":
    print "all works and no play makes me a dull boy!"
'''
