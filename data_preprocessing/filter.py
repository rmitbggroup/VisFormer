import pandas as pd
import json
import csv
import time

output_file_name = '../data/filtered_data.tsv'
max_columns = 25
valid_type = ['line', 'scatter', 'pie', 'bar']
headers = ['fid', 'chart_data', 'layout', 'table_data']
f = open(output_file_name, 'w')
output_file = csv.writer(f, delimiter='\t')
output_file.writerow(headers)

def filter_chunk(chunk, max_column, valid_type, k):
    df_rows = []
    start = time.time()
    num = 0
    for i, x in chunk.iterrows():
        try:
            chart_data = json.loads(x.chart_data)
            layout = json.loads(x.layout)
            table_data = json.loads(x.table_data)  
            
            if not(bool(chart_data) and bool(table_data)):
                continue
                
            fields = table_data[list(table_data.keys())[0]]['cols']
            fields = sorted(fields.items(), key=lambda x:x[1]['order'])
            num_fields = len(fields)
            
            if num_fields > max_column:
                continue
            
            flag = 0
            
            for d in chart_data:
                t = d.get('type')
                if (t == None) or (t not in valid_type):
                    flag = 1
                    break
            
            if flag == 1:
                continue
            
            row = [
                x.fid,
                x.chart_data,
                x.layout,
                x.table_data
                ]
            
            df_rows.append(row)
            num += 1
            
        except Exception as e:
            print (e)
            continue
    output_file.writerows(df_rows)
    print ('chunk: {}, time: {}, rest records: {}'.format(k, time.time()-start, num))

if __name__ == '__main__':
    input_file_name = '../data/sample_data.tsv'
    raw_chunks = pd.read_table(
        input_file_name,
        error_bad_lines = False,
        chunksize = 1000,
        encoding = 'utf-8')
    
    for i, chunk in enumerate(raw_chunks):
        filter_chunk(chunk, max_columns, valid_type, i)
    
    f.close()
    
        
        