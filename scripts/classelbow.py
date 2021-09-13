#Elbow with class pairwise
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
from dataset import dataset_


class ElbowPair(TransformerMixin, BaseEstimator):
    """
    Class of extract dimension from each class pair
    inp: Shrinkage

    """
    def __init__(self, shrinkage=0):
        self.shrinkage=shrinkage


    def fit(self, X, y):
        
        centroid_obj = shrunk_centroid(self.shrinkage)
        df = centroid_obj.create_centroid(X.copy(),y)
        print("Centroid Shape: ", df.shape)
        obj= distance_matrix()
        self.distance_frame = obj.distance(df)

        self.relevant_dims = []
        for pairdistance in self.distance_frame.iteritems():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            #print(pairdistance[0])
            #print(detect_knee_point(distance, indices)[0])
            self.relevant_dims.extend(detect_knee_point(distance, indices)[0])
            self.relevant_dims = list(set(self.relevant_dims))
            #self.relevant_dims = [x for pair in zip(self.relevant_dims,self.relevant_dims) for x in pair]
        
        #print(f"Dimension used% {len(self.relevant_dims)/X.shape[1]}")

        #self.relevant_dims = [0,3,6,7,10,12,13]
        
        return self

    
    def transform(self, X):
        #print(self.relevant_dims)
        #print("Dimension used: ", X.iloc[:, self.relevant_dims].shape[1]/X.shape[1])
        return X.iloc[:, self.relevant_dims]


if __name__ == '__main__':


    #dataset = ['Cricket'] #dataset_
    dataset = ["Epilepsy","EthanolConcentration", "Handwriting", "UWaveGestureLibrary", "AtrialFibrillation", "Libras", "PenDigits"]
    
    for item in dataset:
        print(item)

        #NOTE: Code to read Jump Dataset
        #train_x, train_y = load_from_tsfile_to_dataframe(f"./MP/{item}/TRAIN_X.ts", return_separate_X_and_y=True)
        train_x, train_y = load_from_tsfile_to_dataframe(f"../data/{item}/{item}_TRAIN.ts", return_separate_X_and_y=True)    
        print(f"{item} \nShape: {train_x.shape} ")
        
        obj = ElbowPair(0)
        obj.fit(train_x, train_y)
        print("RS:",obj.relevant_dims)
        df = obj.transform(train_x)
        #print(df.relevant_dims)
        break
    #pass
