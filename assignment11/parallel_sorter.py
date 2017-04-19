#!/usr/bin/env python

'''
Advanced Python assignemtn 11:
    parallel_sorter.py
'''

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_of_process = comm.Get_size()

# rank 0 takes care of generating random data and bucketing
if rank == 0:
    # init random data
    input_size = 10000
    input_arr = np.random.randint(0, input_size, size = input_size)
    
    
    # get bucket range
    min_val = input_arr.min()
    max_val = input_arr.max()
    # make sure bucket range to be at least 1
    num_of_bucket = num_of_process - 1
    bucket_range = (max_val - min_val + num_of_bucket) // num_of_bucket
    
    
    # bucketing
    buckets = dict()
    for i in range(num_of_bucket):
        buckets[i] = list()
    for i in range(input_size):
        v = input_arr[i]
        b_id = (v - min_val) // bucket_range
        buckets[b_id].append(v)
        
        
    # send each bucket to corresponding process for parallel sort 
    for b_id in range(num_of_bucket):
        comm.send(np.array(buckets[b_id], dtype = np.int64), dest=b_id+1)
    
    
    # receiv and reunify sorted buckets
    res = np.empty(0, dtype = np.int64)
    for b_id in range(num_of_bucket):
        buckets[b_id] = comm.recv(source=b_id+1)
        res = np.concatenate((res, buckets[b_id]))
    
    
    # test sorting results
    input_arr.sort()
    np.testing.assert_equal(res, input_arr)
    print("Sorting Done!")

    
# sort one bucket in each other process
else:
    a_bucket = comm.recv(source=0)
    # sort the bucket
    a_bucket.sort()
    # send back to rank 0
    comm.send(a_bucket, dest=0)
    
    
