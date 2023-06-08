import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.8)
ratio = parser.parse_args().ratio

input_file_path = '../data/dataset.tsv'
training_data_path = '../data/training_data.tsv'
test_data_path = '../data/test_data.tsv'

df = pd.read_table(input_file_path,sep='\t')
size = len(df)
training_data = df.iloc[:int(size*ratio)].reset_index(drop=True)
test_data = df.iloc[int(size*ratio)+1:].reset_index(drop=True)

training_data.to_csv(training_data_path,sep='\t',index=False)
test_data.to_csv(test_data_path,sep='\t',index=False)