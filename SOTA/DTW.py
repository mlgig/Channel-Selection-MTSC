import click
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")
import pandas as pd
from multiprocessing import Process
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
import time
from dataset import dataset_

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sktime.transformations.panel.rocket import Rocket
from scripts.classelbow import ElbowPair
from scripts.elbow import elbow
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from scripts.kmeans import kmeans

def agent(path, dataset, folder, datatype,  strategy):

    if datatype=='UCR':
        print(dataset)
        train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")#, return_separate_X_and_y=False)
        test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")#, return_separate_X_and_y=False)

    if strategy == "km":
        elb = kmeans() 
    elif strategy == "ecs":
        elb  = elbow()
    elif strategy == "ecp":
        elb = ElbowPair()
    elif strategy == 'all':
        elb = None

    start = time.time()
    
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
    
    model = Pipeline(
        [

        ('classelbow', elb),
        ('knn', knn),
        ],
        verbose=True,
        )
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    acc1 = accuracy_score(preds, test_y) * 100

    end = time.time()

    results = pd.DataFrame({'Dataset': dataset, 'Accuracy': [acc1], 'Time(min)': [(end - start)/60]})
    print(results)
    temp_path = './'+folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    results.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)

    
@click.command()
@click.option('--datadir', help="Path of datasets", required=True, type=click.Path(exists=True))
@click.option('--tempres', help="Folder to store result", required=True)
@click.option('--datatype', help="UCR data or MP data",  type=click.Choice(['UCR', 'MP'], case_sensitive=True),default="UCR")
@click.option('--strategy', help="KMeans, ECS or ECP",  type=click.Choice(['all','km', 'ecs', 'ecp'], case_sensitive=True),default="ecp")
def cli(datadir, tempres, datatype, strategy):

    #dataset_name = ['Libras']
    dataset_name = dataset_
    processes = []
    for data in dataset_name:

        proc = Process(target=agent, args=(datadir, data, tempres, datatype, strategy))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()
        #agent(datadir, data, tempres, datatype, distancefn, strategy)


if __name__ == '__main__':
    cli()