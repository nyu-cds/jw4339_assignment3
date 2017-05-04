from pyspark import SparkContext
from operator import mul
import re

'''
  Assignment 13 - 2:  
    create an RDD containing the numbers from 1 to 1000， calculates the product 
    of all the numbers from 1 to 1000 and prints the result.
  
  Solution:
    sc.parallelize(range()) to generate the list of number
    then use fold function with mul operator to apply aggreation on the list
'''


if __name__ == '__main__':
    sc = SparkContext("local", "accumulative_multiplication")
    # sc.parallelize(range(1, 1001))  # Generate 1 to 1000 inclusive
    #   .fold(1, mul)              # apply aggreagted operation on the list using mul operator
    print(sc.parallelize(range(1, 1001)).fold(1, mul))
