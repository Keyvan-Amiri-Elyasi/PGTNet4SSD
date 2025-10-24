# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:43:56 2025
@author: Keyvan Amiri Elyasi
"""
import pm4py
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import torch
from torch.utils.data import TensorDataset
from collections import Counter

# TODO: adjust the logic for SSD bucketing
def get_subset_cases(ssd_dict, ssd_id, log, ssd_data_path):
    selected_cases = ssd_dict.get(ssd_id)
    subset_log = pm4py.filter_trace_attribute_values(log, 'concept:name', selected_cases)
    pm4py.write_xes(subset_log, ssd_data_path)    
    return subset_log
 
def create_clusters(X_train, X_val, y_train, y_val, threshold=0.1):
    # Use last timestep
    X_train_2d = X_train[:, -1, :]
    X_val_2d = X_val[:, -1, :]    
    X_combined = np.vstack((X_train_2d, X_val_2d))
    y_combined = np.concatenate((y_train, y_val))
    min_samples = int(threshold * len(X_combined))    
    tree = DecisionTreeRegressor(min_samples_leaf=min_samples, random_state=42)
    tree.fit(X_combined, y_combined)
    return tree

def assign_clusters(tree, X_train, X_val, X_test):
    train_clusters = tree.apply(X_train[:, -1, :])
    val_clusters = tree.apply(X_val[:, -1, :])
    test_clusters = tree.apply(X_test[:, -1, :])    
    clusters_train = train_clusters.tolist() 
    clusters_val = val_clusters.tolist()
    clusters_test = test_clusters.tolist()    
    return clusters_train, clusters_val, clusters_test

def create_partitions(all_lengths, threshold_ratio=0.1):
    # Count occurrences of each prefix length
    length_counts = Counter(all_lengths)
    # Sort by prefix length 
    length_counts = dict(sorted(length_counts.items()))    
    # Compute threshold
    total = sum(length_counts.values())
    min_count = total * threshold_ratio
    partitions = []
    current_range = []
    current_total = 0
    # Loop through lengths in ascending order
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
    # Add any remaining small lengths as final partition
    if current_range:
        partitions.append(current_range)
    unique_lengths = sorted(set(all_lengths))
    merged = sorted([l for group in partitions for l in group])
    assert merged == unique_lengths, "Some lengths are missing or duplicated!"
    return partitions

def get_partition_indices(partition, train_lengths, val_lengths, test_lengths):
    """
    Given a list of lengths (partition) and three datasets' prefix lengths,
    return the indices (positions) in each set that belong to the partition.
    """
    partition_set = set(partition)  # faster membership checking
    train_idx = [i for i, l in enumerate(train_lengths) if l in partition_set]
    val_idx   = [i for i, l in enumerate(val_lengths) if l in partition_set]
    test_idx  = [i for i, l in enumerate(test_lengths) if l in partition_set]
    return train_idx, val_idx, test_idx

def assert_bucketing(train_full, val_full, test_full,
                     train_lst, val_lst, test_lst):
    sum_train = 0
    for lst in train_lst:
        sum_train+=len(lst)
    assert sum_train==len(train_full), 'Bucketing is errounous for train set'
    sum_val = 0
    for lst in val_lst:
        sum_val+=len(lst)
    assert sum_val==len(val_full), 'Bucketing is errounous for validation set'
    sum_test = 0
    for lst in test_lst:
        sum_test+=len(lst)
    assert sum_test==len(test_full), 'Bucketing is errounous for test set'

def conduct_bucketing(X_train, X_val, X_test, 
                      y_train, y_val, y_test,
                      train_lst, val_lst, test_lst,
                      bucket_idx=None,
                      ):
    # select relevant data based on the bucket indices
    idx_tensor = torch.tensor(train_lst[bucket_idx], dtype=torch.long)
    X_train_bucket = X_train[idx_tensor]
    y_train_bucket = y_train[idx_tensor]
    idx_tensor = torch.tensor(val_lst[bucket_idx], dtype=torch.long)
    X_val_bucket = X_val[idx_tensor]
    y_val_bucket = y_val[idx_tensor]
    idx_tensor = torch.tensor(test_lst[bucket_idx], dtype=torch.long)
    X_test_bucket = X_test[idx_tensor]
    y_test_bucket = y_test[idx_tensor]
    # recreate train and validation sets in each bucket 80/20
    X_combined = torch.cat([X_train_bucket, X_val_bucket], dim=0)
    y_combined = torch.cat([y_train_bucket, y_val_bucket], dim=0)
    num_examples = X_combined.size(0)
    split_idx = int(0.8 * num_examples)
    X_train_bucket = X_combined[:split_idx]
    y_train_bucket = y_combined[:split_idx]
    X_val_bucket = X_combined[split_idx:]
    y_val_bucket = y_combined[split_idx:]
    # define training, validation, test datasets                    
    train_dataset = TensorDataset(X_train_bucket, y_train_bucket)
    val_dataset = TensorDataset(X_val_bucket, y_val_bucket)
    test_dataset = TensorDataset(X_test_bucket, y_test_bucket)
    return (train_dataset, val_dataset, test_dataset)