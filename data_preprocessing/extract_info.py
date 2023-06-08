import json
import csv
import pandas as pd
import os
import time

input_file_name = '../data/filtered_data.tsv'
saved_file_path = '../data/processed_data/'
output_file_name = '../data/dataset.tsv'
if not os.path.exists(saved_file_path):
    os.mkdir(saved_file_path)
    
file = open(output_file_name, 'w')
headers = ['fid', 'chart_data', 'layout', 'table_data','num_traces','num_rows','num_fields','pairs','types']
major_type = ['scatter','line','bar','pie']
output_file = csv.writer(file, delimiter='\t')
output_file.writerow(headers)

def check_if_line_chart(d):
    if d.get('mode') in ['lines+markers', 'lines']:
        return True
    if d.get('line') and len(d.get('line').keys()) > 0:
        return True
    if d.get('marker') and 'line' in d.get('marker'):
        return True

    return False

def extract_design_choices(chart_data,layout):
    chart_types = []
    x_srcs = []
    y_srcs = []
    num_traces = 0
    traces = []
    for d in chart_data:
        t = d.get('type')
        if not t or t == 'scatter':
            if check_if_line_chart(d):
                t = 'line'
        
        xsrc = d.get('xsrc')
        if not xsrc:
            xsrc = d.get('labelssrc')
        ysrc = d.get('ysrc')
        if not ysrc:
            ysrc = d.get('valuessrc')
        
        if d.get('orientation') == 'h':
            tmp = xsrc
            xsrc = ysrc
            ysrc = xsrc
        
        cur = [xsrc, ysrc, major_type.index(t)]
        if (not xsrc) and (not ysrc) or (not t):
            continue
        if cur in traces:
            continue
        else:
            traces.append(cur)
            x_srcs.append(xsrc)
            y_srcs.append(ysrc)
            chart_types.append(major_type.index(t))
    
    num_traces = len(chart_types)
    return chart_types, num_traces, x_srcs, y_srcs

def get_data_frame(fields):
    data = {}
    for field in fields:
        data[field[0]] = field[1].get('data')
    return pd.DataFrame(data)

def get_row_num(fields):
    col_len = []
    for field in fields:
        col_len.append(len(field[1].get('data')))
    if len(set(col_len)) > 1:
        return 0
    else:
        return col_len[0]

def get_uids(fields):
    uids = []
    for field in fields:
        uids.append(field[1].get('uid'))
    return uids
    
def extract_table_info(table_data):
    fields = table_data[list(table_data.keys())[0]]['cols']
    fields = sorted(fields.items(), key=lambda x: x[1]['order'])
    num_fields = len(fields)
    
    num_rows = get_row_num(fields)
    if num_rows == 0:
        return None
    #print (num_rows)
    df = get_data_frame(fields)
    uids = get_uids(fields)
    
    return df, num_rows, num_fields, uids
    

def extract_and_save_information(chunk,saved_file_path):
    df_rows = []
    for i, x in chunk.iterrows():
        try:
            fid = x.fid
            chart_data = json.loads(x.chart_data)
            layout = json.loads(x.layout)
            table_data = json.loads(x.table_data)
                
            if extract_design_choices(chart_data,layout):
                chart_types, num_traces, x_srcs, y_srcs = extract_design_choices(chart_data,layout)
            else:
                continue
            
            if extract_table_info(table_data):
                df, num_rows, num_fields, uids = extract_table_info(table_data)
            else:
                continue
                
            pairs = []
            
            for xsrc, ysrc in list(zip(x_srcs,y_srcs)):
                if xsrc and ysrc:
                    pairs.append([uids.index(xsrc.split(':')[-1]), uids.index(ysrc.split(':')[-1])])
                elif xsrc == None:
                    pairs.append([num_fields,uids.index(ysrc.split(':')[-1])])
                elif ysrc == None:
                    pairs.append([uids.index(xsrc.split(':')[-1]),num_fields])
            
            fid = '_'.join(fid.split(':'))
                
            saved_chart_data_path = saved_file_path+'chart_data'+'/'+fid+'.json'
            saved_table_data_path = saved_file_path+'table_data'+'/'+fid+'.tsv'
            saved_layout_data_path = saved_file_path+'layout'+'/'+fid+'.json'
                
                #Write chart data json file
            with open(saved_chart_data_path ,'w') as f:
                json.dump(chart_data,f)
                #Write table data csv file
            df.to_csv(saved_table_data_path, index = False)
                #Write layout data json file
            with open(saved_layout_data_path,'w') as f:
                json.dump(layout,f)
            df_rows.append([
                fid,
                saved_chart_data_path,
                saved_layout_data_path,
                saved_table_data_path,
                num_traces,
                num_rows,
                num_fields,
                pairs,
                chart_types]
                )         
        except Exception as e:
            continue
            
    output_file.writerows(df_rows)               

if __name__ == '__main__':
    for s in ['table_data','layout','chart_data']:
        if not os.path.exists(saved_file_path+s):
            os.mkdir(saved_file_path+s)
    
    raw_chunks = pd.read_table(
        input_file_name,
        error_bad_lines = False,
        chunksize = 1000,
        encoding = 'utf-8'
        )
    
    for i, chunk in enumerate(raw_chunks):
        start = time.time()
        extract_and_save_information(chunk, saved_file_path)
        print ('Chunk {}: Processed Time:{} seconds'.format(i, time.time()-start))
    file.close()
        
                
            
            
            
            
            
            