from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random
import itertools
import numbers

from savecsv import preds_to_csv



class Node:
    def __init__(self, feature=None, thresh=None, label=None, left=None, right=None):
        self.feature = feature
        self.thresh = thresh
        self.label = label
        self.left = left
        self.right = right


class DecisionTree:

    def __init__(self, max_depth=15, header=''):
        self.train_tree = None
        self.max_depth = max_depth
        self.header = header
        
    @staticmethod
    def entropy(y):
        _, counts = np.unique(y,return_counts=True)
        return stats.entropy(counts, base=2)

    @staticmethod
    def information_gain(X, y_left, y_right):
        entropy_pre = DecisionTree.entropy(X)
        len_yl, len_yr = len(y_left), len(y_right)
        entropy_after = (len_yl*DecisionTree.entropy(y_left) + len_yr*DecisionTree.entropy(y_right))/(len_yl + len_yr)
        return entropy_pre - entropy_after

    @staticmethod
    def split(X, y, feature, thresh):
        left_filter = np.where(X[:, feature]>thresh)[0]
        right_filter = np.where(X[:, feature]<= thresh)[0]
        return X[left_filter], y[left_filter], X[right_filter], y[right_filter]
    
    @staticmethod
    def best_split(x, y):
        best_gain, best_thresh, best_feature = 0,0,0
        num_features = x.shape[1]
        features = range(num_features)
        threshs = [value for feature in features for value in np.unique(x[:,feature])]
        results = []
        for feature, value in itertools.product(features, threshs):
            _, y_left, _, y_right = DecisionTree.split(x,y,feature, value)
            if len(y_left) > 0 and len(y_right) > 0:
                info_g = DecisionTree.information_gain(y,y_left,y_right)
                if info_g > best_gain:
                    best_gain = info_g
                    best_thresh, best_feature = value, feature
        return best_feature, best_thresh

    def grow_tree(self, x, y, depth=0):
        majority_value = Counter(y).most_common(1)[0][0] # return the most common label
        if DecisionTree.entropy(y) == 0 or len(y)<300 or depth == self.max_depth:
            return Node(label=majority_value)
        feature, thresh = DecisionTree.best_split(x,y)
        x_left, y_left, x_right, y_right = DecisionTree.split(x,y,feature,thresh)
        if len(y_left)>0 and len(y_right)>0:
            left = self.grow_tree(x_left,y_left,depth+1)
            right = self.grow_tree(x_right,y_right,depth+1)
            label = "feature #" + str(feature) + ">" + str(thresh)
            return Node(feature, thresh, label, left, right)
        else:
            return Node(label=majority_value)
    
    def fit(self, X, y):
        self.train_tree = self.grow_tree(X,y)
    
    def predict_single(self, node, sample):
        cur_node = node
        while not isinstance(cur_node.label, numbers.Number):
            cur_node = cur_node.left if sample[cur_node.feature] > cur_node.thresh else cur_node.right
        return cur_node.label

    def predict(self, x, mode='train', tree_type='decisiontree',rf=None):
        num_samples = x.shape[0]
        y = np.zeros((num_samples))
        for row in range(num_samples):
            pred = None
            if tree_type == 'decisiontree':
                pred = self.predict_single(self.train_tree, x[row,:])
            elif tree_type == 'randomforest':
                pred = self.predict_single(rf, x[row,:])
            y[row] = pred
        if mode == "test" and tree_type == "decisiontree":
                preds_to_csv(y,self.header)
        else:
            return y

    def print_tree(self):
        def print_level(node):
            depth = 0
            cur_level = [node]
            while cur_level:
                next_level = []
                for node in cur_level:
                    print("Depth", str(depth)+":",str(node.label))
                    if node.left:
                        next_level.append(node.left)
                    if node.right:
                        next_level.append(node.right)
                depth += 1
                cur_level = next_level
                print()
        print_level(self.train_tree)
    


class RandomForest(DecisionTree):
    
    def __init__(self,num_trees=0,rand_features=None,max_depth=15,header=""):
        super(RandomForest, self).__init__()
        self.trained_trees = [None] * num_trees
        self.num_trees = num_trees
        self.rand_features = "sqrt"
        self.max_depth = max_depth  
        self.header = header

   
    def parallel_grow_tree(self, x, y):
        fsize = None
        if self.rand_features == "sqrt":
            fsize = int(np.sqrt(x.shape[1]))
        elif self.rand_features == "third":
            fsize = x.shape[1]//3
        elif self.rand_features == "log":
            fsize = int(np.log2(x.shape[1]+1))
        feature_indices = np.random.choice([i for i in range(x.shape[1])],\
            size=fsize,replace=False)
        sample_indices = np.random.randint(x.shape[0], size=x.shape[0])
        x = x[:,feature_indices]
        x = x[sample_indices, :]
        return super(RandomForest, self).grow_tree(x, y)

    def fit(self, x, y):
        self.trained_trees = [self.parallel_grow_tree(x,y) for _ in range(self.num_trees)]

    def parallel_predict(self, x, model):
        return super(RandomForest, self).predict(x,tree_type="randomforest",rf=model)
    
    def predict(self, x, mode="train"):
        all_preds = np.asarray(np.matrix([self.parallel_predict(x,self.trained_trees[i]) for i in range(self.num_trees)]))
        final_preds = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, all_preds)
        if mode == "test":
            preds_to_csv(final_preds,self.header)
        else:
            return final_preds