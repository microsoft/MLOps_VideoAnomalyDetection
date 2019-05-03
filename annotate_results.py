import pandas as pd
import os
import argparse

# Define args
parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--data_dir', default='./data/video/', type=str, dest='data_dir', help='path to data and annotations (annotations should be in <data_dir>/<dataset>/Test/<dataset>.m')
parser.add_argument('--dataset', default='UCSDped1', type=str, dest='dataset', help='dataset we are using')
parser.add_argument('--nt', default=200, type=int, dest='nt', help='length of video sequences')

# Parse args
args = parser.parse_args()
data_dir = args.data_dir
dataset = args.dataset
nt = args.nt

# extent data_dir for current dataset
data_dir = os.path.join(data_dir, dataset, 'Test')

# Load the result of fitting the trained model to the test data
df = pd.read_pickle(os.path.join(data_dir, 'test_results.pkl.gz'))

# The next step is to merge that with the labels for anomalies in the ucsd dataset 
# UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/UCSDped1.m
anom_anot_filename = os.path.join(data_dir, '%s.m' % dataset)
with open(anom_anot_filename, 'r') as f:
    lines = f.readlines()
del lines[0] # remove file header

# extract the beginning and end of subsequences that contain anomalies
anom_indices = []
for l, line in enumerate(lines):
    line = line.replace(":", ",")
    anom_indices.append(line.split("[")[1].split("]")[0].split(","))

# add column with anomalies to df
df['anom'] = 0
for a, anom in enumerate(anom_indices):
    while len(anom) > 0:
        first_frame = int(anom.pop(0)) + a * nt
        last_frame = int(anom.pop(0)) + a * nt
        print("First/last Frame of Anomaly: %s/%s" % (first_frame, last_frame))
        df.loc[first_frame:last_frame, 'anom'] = 1

# save the dataframe 
df.to_pickle(os.path.join(data_dir, '%s.pkl.gz' % dataset))
