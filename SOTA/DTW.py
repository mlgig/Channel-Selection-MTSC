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

def agent(path, dataset, seg, folder,  paa=True):

    if dataset in dataset_: #['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']:
        train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")#, return_separate_X_and_y=False)
        test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")#, return_separate_X_and_y=False)
    
    elif dataset in ['FullUnnormalized', 'Normalized', 'Unnormalized']:
        train_x, train_y = load_from_tsfile_to_dataframe(f"./MP/{dataset}/TRAIN_X.ts")
        test_x, test_y = load_from_tsfile_to_dataframe(f"./MP/{dataset}/TEST_X.ts")#, return_separate_X_and_y=False)
    #print(f"{dataset}:Before Train Shape {train_x.shape}")
    #print(f"{dataset}:Before Test Shape {test_x.shape}")

    start = time.time()
    elb = elbow()#kmeans()#ElbowPair(0)#elbow()# ElbowPair(0) #kmeans() #ElbowPair
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
    model = Pipeline(
        [

        #('kmeans', elb),
        #('classelbow', elb),
        ('elbow', elb),
        #('data_transform', PAAStat(paa_=paa, seg_=seg)),
        ('knn', knn),
        #('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True, class_weight='balanced' ))
        ],
        verbose=True,
        #memory="./cache"
        )
        

    model.fit(train_x, train_y)
    #print(elb.relevant_dims)
    preds = model.predict(test_x)
    acc1 = accuracy_score(preds, test_y) * 100
    f1_mic = f1_score(preds, test_y, average='micro')
    f1_mac = f1_score(preds, test_y, average='macro')

    end = time.time()

    #results = pd.DataFrame({'Dataset': dataset, 'Accuracy_ED': [acc1], 'Time(min)': [(end - start)/60]})


    results = pd.DataFrame({'Dataset': dataset, 'Accuracy': [acc1], 'f1_macro': [f1_mac], 'f1_micro': [f1_mic], 
    
    'Time(min)': [(end - start)/60], 
    'dimension_used':[len(elb.relevant_dims)/train_x.shape[1]]
    })
    print(results)

    #print(results)
    #print("Value Counts: ", pd.Series(test_y).value_counts())
    #print("Confusion Matrix")
    #print(pd.DataFrame(confusion_matrix( x_test, preds, labels=pd.Series(test_y).unique())))
    #cm = plot_confusion_matrix(model,x_test, test_y, normalize='true')
    #plt.show()
    #plt.savefig(f'./notebooks/MP_images/{dataset}_CM_CD_T2.png')

    temp_path = './'+folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    results.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)

    
    
@click.command()
@click.option('--path', help="Path of datasets", required=True, type=click.Path(exists=True))
@click.option('--paa', help="PAA", type=click.Choice(['True', 'False'], case_sensitive=True))
@click.option('--folder', help="Folder to store result", required=True)
@click.option('--seg', help="compression ratio", required=True, type=float, nargs=3)
def cli(path, paa, folder, seg):

    #dataset_name = UCR_UEA_datasets().list_multivariate_datasets()
    #dataset_name = sorted([dataset[1] for dataset in datasets if len(dataset[1])>=1][0])
    dataset_name = dataset_
    print(dataset_name)
    processes = []
    for data in dataset_name:
        print("Dataset: ", data)
        #agent(path, data, seg, folder, paa)
        proc = Process(target=agent, args=(path, data, seg, folder, paa))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    cli()
