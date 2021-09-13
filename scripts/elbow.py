#ElbowCut
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from scripts.shrunk_cent import shrunk_centroid
from scripts.calc_distance import distance_matrix
from scripts.utils import detect_knee_point
from sklearn.base import TransformerMixin, BaseEstimator

class elbow(TransformerMixin, BaseEstimator):

    def fit(self, X, y):
        
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(),y)
        obj= distance_matrix()
        self.distance_frame = obj.distance(df)

        self.relevant_dims = []
        distance = self.distance_frame.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        #print(detect_knee_point(distance, indices))
        self.relevant_dims.extend(detect_knee_point(distance, indices)[0])
        #print(distance)
        
        return self

    
    def transform(self, X):
        #print(self.relevant_dims)
        #print("Dimension used: ", X.iloc[:, self.relevant_dims].shape[1]/X.shape[1])
        return X.iloc[:, self.relevant_dims]


if __name__ == '__main__':
    
    dataset = ['AtrialFibrillation']
    for item in dataset:
        print(item)

        #NOTE: Code to read Jump Dataset
        #train_x, train_y = load_from_tsfile_to_dataframe(f"./MP/{item}/TRAIN_X.ts", return_separate_X_and_y=True)
        train_x, train_y = load_from_tsfile_to_dataframe(f"../data/{item}/{item}_TRAIN.ts", return_separate_X_and_y=True)
        #print(f"{item} \nShape: {train_x.shape} ")
        
        obj = elbow()
        obj.fit(train_x, train_y)
        print("RS: ",obj.relevant_dims)
        df = obj.transform(train_x)
        print(obj.distance_frame())
        print(df.shape)
    #pass
