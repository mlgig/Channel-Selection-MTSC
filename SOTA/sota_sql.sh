#Activate the environment
source ~/mtsc/bin/activate
#Benchmarking SOTA-WM
#python3 -W ignore -u weasel_muse.py --path ../data --paa True --folder centroid --seg 0.30 0.6 0.9 >> weasellog.txt
#python3 ../../Generic_Scripts/combine_csv.py ./centroid weasel_muse


python3 -W ignore -u MrSEQLMain.py --path ../data --paa True --folder centroiddr --seg 0.30 0.6 0.9 >> seqlred.txt
#python3 ../../Generic_Scripts/combine_csv.py ./centroid weasel_muse
