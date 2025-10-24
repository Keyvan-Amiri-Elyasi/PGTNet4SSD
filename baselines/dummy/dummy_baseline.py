# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 08:46:44 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import pickle
import pm4py

def split_log(log, train_ratio=0.8, 
              case_col ='case:concept:name', time_col='time:timestamp'):
    case_start_times = log.groupby(case_col)[time_col].min()
    sorted_case_ids = case_start_times.sort_values().index.tolist()
    split_index = int(len(sorted_case_ids) * train_ratio)
    train_case_ids = sorted_case_ids[:split_index]
    test_case_ids = sorted_case_ids[split_index:]
    train_df = log[log[case_col].isin(train_case_ids)].copy()
    test_df = log[log[case_col].isin(test_case_ids)].copy()
    return (train_case_ids, test_case_ids, train_df, test_df)

def add_rem_time(df_inp,
                 case_col ='case:concept:name', time_col='time:timestamp'):
    df = df_inp.copy()
    df['rem_time'] = df.groupby(case_col)[time_col].transform(
        lambda x: (x.max() - x).dt.total_seconds()
        )
    return df

def add_prefix_length(df_inp,
                      case_col ='case:concept:name', time_col='time:timestamp'):
    df = df_inp.copy()
    df = df.sort_values([case_col, time_col])
    df['prefix_length'] = df.groupby(case_col).cumcount() + 1
    return df

def create_length_partitions(df_inp, threshold_ratio=0.1,
                             case_col ='case:concept:name'):
    df1 = df_inp[df_inp.groupby(case_col)['prefix_length'].transform('max') 
                      != df_inp['prefix_length']].copy()
    df = df1[df1['prefix_length'] != 1].copy()  
    length_counts = df['prefix_length'].value_counts().sort_index()
    min_count = len(df) * threshold_ratio
    partitions = []
    current_range = []
    current_total = 0    
    for length, count in length_counts.items():
        if count >= min_count:
            if current_range:
                partitions.append(current_range)
                current_range = []
            partitions.append([length])
        else:
            current_range.append(length)
            current_total += count
            if current_total >= min_count:
                partitions.append(current_range)
                current_range = []
                current_total = 0    
    # Add all remaining lengths as one final partition
    if current_range:
        partitions.append(current_range)    
    return partitions

def find_closest_partition_index(length, partitions):
    return min(range(len(partitions)), key=lambda i: min(abs(length - x) for x in partitions[i]))

def no_bucket_dummy(train_df, test_df, case_col ='case:concept:name'):
    train1 = train_df[train_df.groupby(case_col)['prefix_length'].transform('max') 
                      != train_df['prefix_length']].copy()
    sel_train = train1[train1['prefix_length'] != 1].copy()    
    mean_rem = sel_train['rem_time'].mean()
    test1 = test_df[test_df.groupby(case_col)['prefix_length'].transform('max') 
                      != test_df['prefix_length']].copy()
    sel_test = test1[test1['prefix_length'] != 1].copy()    
    mae_test = (sel_test['rem_time'] - mean_rem).abs().mean()/3600/24
    return mae_test

def length_bucket_dummy(train_df, test_df, partitions,
                        case_col ='case:concept:name'):
    train1 = train_df[train_df.groupby(case_col)['prefix_length'].transform('max') 
                      != train_df['prefix_length']].copy()
    sel_train = train1[train1['prefix_length'] != 1].copy()  
    mean_lst = []
    for partition in partitions:
        df = sel_train[sel_train['prefix_length'].isin(partition)]
        mean_rem = df['rem_time'].mean()
        mean_lst.append(mean_rem)
    test1 = test_df[test_df.groupby(case_col)['prefix_length'].transform('max') 
                          != test_df['prefix_length']].copy()
    sel_test = test1[test1['prefix_length'] != 1].copy() 
    sel_test['prediction'] = sel_test['prefix_length'].apply(
    lambda x: mean_lst[find_closest_partition_index(x, partitions)])
    mae_test = (sel_test['prediction'] - sel_test['rem_time']).abs().mean() / (24 * 3600)    
    return mae_test    

def main():
    # TODO: adjust the data directory (in github repo)
    parser = argparse.ArgumentParser(
        description='DUMMY model for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--bucketing', type=str, default='L_B',
                        choices=['N_B', 'L_B', 'C_B', 'SSD_B'],
                        help='Bucketing Strategy to be used.')
    args = parser.parse_args()
    data_dir = r'C:\SNA-data\data'
    dataset_path = os.path.join(data_dir,args.dataset)
    log = pm4py.read_xes(dataset_path)
    df = add_rem_time(log)
    df = add_prefix_length(df)
    (train_case_ids, test_case_ids, train_df, test_df) = split_log(df)
    if args.bucketing == 'N_B':
        mae_test = no_bucket_dummy(train_df, test_df)
    elif args.bucketing == 'L_B':
        partitions = create_length_partitions(train_df)
        mae_test = length_bucket_dummy(train_df, test_df, partitions)
    elif args.bucketing == 'C_B':
        print('Clustering based bucketing cannot be applied to DUMMY model.')
    else:
        #TODO: add steady-state logic here
        pass
    print(mae_test)
  
    """
    root_path = os.getcwd()
    file_name = 'simulated_log_4_1_SS_classes.pkl'
    file_path = os.path.join(root_path,file_name)
    ssd_dict = pickle.load(open(file_path, "rb"))
    for key in ssd_dict:
        print(key, len(ssd_dict[key]))
    """
    

if __name__ == '__main__':
    main()  