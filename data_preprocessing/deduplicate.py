import pandas as pd

input_file_path = '../data/dataset.tsv'
output_file_path = '../data/dataset.tsv'

df = pd.read_table(input_file_path, sep='\t')

uids = []
fids = df['fid']

for fid in fids:
    uids.append('_'.join(fid.split('_')[:-1]))

#df.insert(loc=0, column='uid', value=uids)
subset = ['uid', 'num_traces', 'num_fields', 'num_rows', 'pairs', 'types']
df.drop(df[df['num_traces']==0].index,inplace=True)
df.drop_duplicates(subset=subset, keep='first',inplace=True)

idxes = []
for i in range(len(df)):
    table_path = df.iloc[i]['table_data']
    table = pd.read_csv(table_path, sep=',')
    if len(table.columns) != df.iloc[i]['num_fields']:
        idxes.append(i)
df.drop(idxes, inplace = True)     
df.to_csv(output_file_path, index=False, sep='\t')

