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
pd.set_option('display.max_columns', None)  
#pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sktime.transformations.panel.rocket import Rocket
from scripts.classelbow import ElbowPair
from itertools import chain, combinations

def agent(path, dataset, seg, folder,  paa=True):

    if dataset in dataset_: #['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']:
        train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")#, return_separate_X_and_y=False)
        test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")#, return_separate_X_and_y=False)
    
    elif dataset in ['FullUnnormalized', 'Normalized', 'Unnormalized']:
        train_x, train_y = load_from_tsfile_to_dataframe(f"./MP/{dataset}/TRAIN_X.ts")
        test_x, test_y = load_from_tsfile_to_dataframe(f"./MP/{dataset}/TEST_X.ts")#, return_separate_X_and_y=False)
#    #print(f"{dataset}:Before Train Shape {train_x.shape}")
#    #print(f"{dataset}:Before Test Shape {test_x.shape}")
#
##Create the subsets here of the dimensions
    #print(np.arange(train_x.shape[1]))
    
    s = np.arange(train_x.shape[1])
    dims_subset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    results = pd.DataFrame({'Dataset': [dataset]})
    acc1=0
    for item in dims_subset:


        model = Pipeline(
        [

        #('kmeans', elb),
        #('classelbow', elb),
        #('elbow', elb),
        #('data_transform', PAAStat(paa_=paa, seg_=seg)),
        ('rocket', Rocket(random_state=0)),
        ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True, class_weight='balanced' ))
        ],
        #verbose=True,
        #memory="./cache"
        )


        if len(list(item))>=1:
            #print(list(item))
            #print(train_x.iloc[:,list(item)].shape)

            model.fit(train_x.iloc[:,list(item)], train_y)
            #print(elb.relevant_dims)
            preds = model.predict(test_x.iloc[:,list(item)])
            #print("XX", acc1, item)
            tmp = accuracy_score(preds, test_y) * 100
            if acc1<tmp:
                print(f"ACC:{tmp}, ITEM: {item}")
                acc1 = tmp    
                _acc = pd.DataFrame({f'Accuracy': [tmp],


                #'Time(min)': [(end - start)/60], 
                'dimension':str(item)
                })
                results = _acc #pd.concat([results, _acc], axis=1)

            #print(results)
            del model

    print(results)
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

    #dataset_name = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset_name = ['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF','FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset_name = ['Unnormalized']
    dataset_name = dataset_
    #dataset_name = ['Cricket']
    print(dataset_name)
    for data in dataset_name:
        print("\n",data)
        agent(path, data, seg, folder, paa)
        #break

        

if __name__ == '__main__':
    #path = '../mtsc/data/'
    #paa = True
    #folder = 'centroid_50'
    #seg = "0.30 0.6 0.9"
    #cli(path, paa, folder, seg)
    cli()
    

#python3 -W ignore main.py -- -- --folder centroid_50 --
