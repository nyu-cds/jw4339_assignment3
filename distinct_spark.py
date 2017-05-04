from pyspark import SparkContext
from operator import add
import re

'''
  Assignment 13 - 1:   count the number of distinct words in the input text
  
  Solution:
    During reduction, emit every single unique word, with 1 number of partition
    Then use the count method of RDD
'''


# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
    sc = SparkContext("local", "word_count_distinct")
    
    text = sc.textFile('pg2701.txt')
    words = text.flatMap(splitter)
    words_mapped = words.map(lambda x: (x,1))

    # During reduction, emit every unique word once
    words_mapped_uniq = words_mapped.reduceByKey(lambda x, y: 1, numPartitions=1)
    # print(words_mapped_uniq.sortByKey().take(100))
    print(words_mapped_uniq.count())