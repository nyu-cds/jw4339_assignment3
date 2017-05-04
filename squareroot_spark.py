from pyspark import SparkContext
from operator import add
from math import sqrt

'''
  Assignment 13 - 3:  
    use map and fold to calculate the average of the square root of all the 
    numbers from 1 to 1000. i.e the sum of the square roots of all the 
    numbers divided by 1000.
  
  Solution:
    1. generate RDD from the list of 1 to 1000
    2. use map to get all square root, save into RDD namded sqrts
    3. cache the sqrts for later fold operation
    4. apply folding add on sqrts to get the sum of all square root
    5. calculate avg = sum_sqrts / 1000, and print it out
'''


if __name__ == '__main__':
    sc = SparkContext("local", "sqrt_1000")

    # generate
    nums = sc.parallelize(range(1, 1001))
    sqrts = nums.map(sqrt)
    sqrts.cache()
    sum_sqrts = sqrts.fold(0, add)
    print(sum_sqrts / 1000)
