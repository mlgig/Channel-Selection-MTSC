import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append("..")
import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from scripts.calculate_centroid import centroid
from scripts.utils import *
from operator import truediv 
from sktime.utils.data_processing import from_nested_to_3d_numpy, from_3d_numpy_to_nested
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
#from fastdtw import fastdtw


class shrunk_centroid:

    def __init__(self, shrink):
        self.shrink = shrink
    
    def create_centroid(self, X, y):
        """
        Creating the centroid for each class
        """

        #y = X.class_vals
        #X.drop('class_vals', axis = 1, inplace = True)
        cols = X.columns.to_list()   
        ts = from_nested_to_3d_numpy(X) # Contains TS in numpy format
        centroids = []

        le = LabelEncoder()
        y_ind = le.fit_transform(y)

        for dim in range(ts.shape[1]):
            train  = ts[:, dim, :]
            clf = NearestCentroid(train)
            clf = NearestCentroid(shrink_threshold = self.shrink)
            clf.fit(train, y)
            centroids.append(clf.centroids_)

        centroid_frame = from_3d_numpy_to_nested(np.stack(centroids, axis=1), column_names=cols)
        centroid_frame['class_vals'] = clf.classes_ 
        
        return centroid_frame.reset_index(drop =True)

if __name__ == "__main__":
    
    #jump_dataset = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset = ['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']
    dataset = ['Cricket']
    for item in dataset:
        print(item)
        #print(f"./data/{item}/{item}_TRAIN.ts")
        #train_x, y = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts",return_separate_X_and_y=True)
        train_x = load_from_tsfile_to_dataframe("./data/Cricket/Cricket_TRAIN.ts",return_separate_X_and_y=False)
        print(train_x.shape)
        obj = shrunk_centroid(0)
        df_s = obj.create_centroid(train_x.copy())
        print(df_s.shape)
        print(train_x.shape)
    
