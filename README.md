# Fast Channel Selection for Scalable Multivariate Time Series Classification
Code Repo for paper AALTD@ECMLPKDD21 Paper

## Abstract
Multivariate time series record sequences of values using multiple sensors or channels. In the classification task, we have a class label associated with each multivariate time series. For example, a smartwatch captures the activity of a person over time, and there are typically multiple sensors capturing aspects of motion such as acceleration, orientation, heart beat. Existing Multivariate Time Series Classification (MTSC) algorithms do not scale well with large datasets, and this leads to extensive training and prediction times. This problem is attributed to an increase in the number of records (e.g., study participants), duration of recording (time series length), and number of channels (e.g., sensors). 
Existing MTSC methods do not scale well with the number of channels, and only a few  methods can complete their training on the medium sized UEA MTSC benchmark within 7 days. Additionally, for some problems, only a few channels are relevant for the learning task, and thus identifying the relevant channels before training may help with improving both the scalability and accuracy of the classifiers, as well as result in savings for data collection and storage.
In this work, we investigate a few channel selection strategies for MTSC and propose a new approach for fast supervised channel selection. The key idea is to use channel-wise class separation estimation using fast computation on centroid-pairs. We evaluate the impact of our new method on the accuracy and scalability of a few state-of-the-art MTSC algorithms and show that our approach can dramatically reduce the input data size, and thus improve scalability, while also preserving accuracy. In some cases, the runtime for training the classifier was reduced to one third of the runtime on the original dataset. We also analyse the performance of our channel selection method in a case study on a human motion classification task and show that we can achieve the same accuracy using only one third of the data.

## Results
Total time taken by three channel selection strategies on 26 UEA datasets.

![image](https://user-images.githubusercontent.com/20501023/127819441-1335ad5b-3b11-47f5-a1d9-41d3f7ddfa3f.png)

Loss/Gain in mean accuracy (∆ Acc) vs percentage time saved (% Time) with respect to All channels for our three channel selection techniques on 26 UCR datasets. The red and blue color indicates loss and gain in accuracy respectively. Higher value for % Time or % Storage indicates more time or storage saved.


![image](https://user-images.githubusercontent.com/20501023/127819519-dfae8b4f-9d46-4c98-bd0a-36331cfdb410.png)


## Datasets
Download and unzip the mtsc data from UCR-UEA archive to data folder. Uncomment the data for which results need to be obtained in `dataset.py` file.


## Running Experiments

`python3 -W ignore -u SOTA/file_name.py --help` <br>
`python3 -W ignore -u SOTA/file_name.py --datadir data --tempres temp  --strategy ecs`


## Citation

```
@inproceedings{dhariyal2021fast,
  title={Fast Channel Selection for Scalable Multivariate Time Series Classification},
  author={Dhariyal, Bhaskar and Nguyen, Thach Le and Ifrim, Georgiana},
  booktitle={International Workshop on Advanced Analytics and Learning on Temporal Data},
  pages={36--54},
  year={2021},
  organization={Springer}
}
```

## Disclaimer

These software distributions are open source, licensed under the GNU General Public License (v3 or later).
Note that this is the full GPL, which allows many free uses, but does not allow its incorporation
(even in part or in translation) into any type of proprietary software which you distribute.
Commercial licensing is also available; please contact us if you are interested.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

