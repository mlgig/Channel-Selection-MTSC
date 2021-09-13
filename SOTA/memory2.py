import click
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")
import pandas as pd
from dataset import dataset_
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from scripts.classelbow import ElbowPair
from scripts.elbow import elbow
from scripts.kmeans import kmeans

def agent(path, dataset):

    #start = time.time()
    train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")#, return_separate_X_and_y=False)
    #train_x, train_y = load_from_tsfile_to_dataframe("/home/bhaskar/Desktop/CentroidMTSC/dataset/Openpose/Normalized/TRAIN_X.ts")
    test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")#, return_separate_X_and_y=False)
    #test_x, test_y = load_from_tsfile_to_dataframe("/home/bhaskar/Desktop/CentroidMTSC/dataset/Openpose/Normalized/TEST_X.ts")#, return_separate_X_and_y=False)
    #print(f"Function {fn}")
    #print(f"{dataset}:Before Train Shape {train_x.shape}")
    print(f"Original: {train_x.memory_usage(deep=True).sum()/1024/1024} MB")
    #print(f"{dataset}:Before Test Shape {test_x.shape}")
    
    original = train_x.memory_usage(deep=True).sum()
    elb = ElbowPair(0) #kmeans() # elbow()#kmeans() #ElbowPair(0) # ElbowPair 
    
    elb.fit(train_x, train_y)
    #print("Relevant Dims :",elb.relevant_dims)
    reduced = train_x.iloc[:, elb.relevant_dims].memory_usage(deep=True).sum() 
    print(f"Reduced: {train_x.iloc[:, elb.relevant_dims].memory_usage(deep=True).sum()/1024/1024} MB"  )

    print(f"Percentage Red: {(original-reduced)*100/original}")

    print("--"*50)

    res = pd.DataFrame({'Dataset': dataset, 'OriMB': [original], 'RedMB' :[reduced]})
    temp_path = "./mem"
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    res.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)

@click.command()
@click.option('--path', help="Path of datasets", required=True, type=click.Path(exists=True))
def cli(path):

    dataset_name = dataset_
    #dataset_name = ['AtrialFibrillation']
    #print(dataset_name)
    for data in dataset_name:
        print(data)
        agent(path, data)
        #break


if __name__ == '__main__':
    #path = '../mtsc/data/'
    #paa = True
    #folder = 'centroid_50'
    #seg = "0.30 0.6 0.9"
    #cli(path, paa, folder, seg)
    cli()
