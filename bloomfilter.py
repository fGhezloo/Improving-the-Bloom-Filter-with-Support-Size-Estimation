# Python 3 program to build Bloom Filter 
# Install mmh3 and bitarray 3rd party module first 
# pip install mmh3 
# pip install bitarray 
import math 
import mmh3 
from bitarray import bitarray
import numpy as np
  
class BloomFilter(object): 
  
    ''' 
    Class for Bloom filter, using murmur3 hash function 
    '''
  
    def __init__(self, items_count, fp_prob):
        ''' 
        items_count : int 
            Number of items expected to be stored in bloom filter 
        fp_prob : float 
            False Positive probability in decimal 
        '''
        # False posible probability in decimal 
        self.fp_prob = fp_prob 
        
        self.items_count = items_count
        # Size of bit array to use 
        self.size = self.get_size(items_count, fp_prob)

        # number of hash functions to use 
        self.hash_count = self.get_hash_count(self.size,items_count) 
  
        # Bit array of given size 
        self.bit_array = bitarray(self.size) 
  
        # initialize all bits as 0 
        self.bit_array.setall(0) 
  
    def add(self, item): 
        ''' 
        Add an item in the filter 
        '''
        digests = [] 
        for i in range(self.hash_count): 
  
            # create digest for given item. 
            # i work as seed to mmh3.hash() function 
            # With different seed, digest created is different 
            digest = mmh3.hash(item,i) % self.size 
            digests.append(digest) 
  
            # set the bit True in bit_array 
            self.bit_array[digest] = True
  
    def check(self, item): 
        ''' 
        Check for existence of an item in filter 
        '''
        for i in range(self.hash_count): 
            digest = mmh3.hash(item,i) % self.size 
            if self.bit_array[digest] == False: 
  
                # if any of bit is False then,its not present 
                # in filter 
                # else there is probability that it exist 
                return False
        return True
  
    @classmethod
    def get_size(self,k,p):
        ''' 
        Return the size of bit array(m) to used using 
        following formula 
        b = -(k * lg(p)) / (lg(2)^2)
        k : int
            number of items expected to be stored in filter 
        p : float 
            False Positive probability in decimal 
        '''
        b = -(k * math.log(p))/(math.log(2)**2)
        return int(b)
  
    @classmethod
    def get_hash_count(self, b, k):
        ''' 
        Return the hash function(k) to be used using 
        following formula 
        h = (b/k) * lg(2)
  
        b : int 
            size of bit array 
        k : int
            number of items expected to be stored in filter 
        '''
        h = (b/k) * math.log(2)
        return int(h)
    
    def memory(self):
        return self.hash_count * self.size

    def trueFP(self, k):
        """If there were actually k objects hashed, what would the FP
        rate of this filter be?"""
        return (1 - np.exp(-self.hash_count * k / self.size))**self.hash_count
