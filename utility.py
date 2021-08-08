import csv
from bloomfilter import BloomFilter
from GoodTuring import goodTuringEst
import numpy as np


class utility:
    '''
    A pipeline for testing our code.
    '''

    def __init__(self, data, fp, n, estimator, unseenEst=None):
        """Create a testing object based on the list data,
        with nominal false positive rate fp, and n initial samples,
        where the support size estimation uses the estimator"""
        self.fp = fp
        self.n = n
        self.entireFile = data
        self.estimator = estimator
        self.unseenEst = unseenEst

        self.K = len(data)

        self.samplePool = data[0:n]

        self.kHat = self.getSupportSizeEst()

        self.bf = self.initBloomFilter(self.kHat, self.fp, self.samplePool)

    def getTrueFP(self):
        # How many distinct objects are in the file?
        numItems = len(np.unique(self.entireFile))
        return self.bf.trueFP(numItems)

    def getSize(self):
        """Return the size of the bloom filter"""
        return self.bf.memory()

    def getOptSize(self):
        # How many distinct objects are in the file?
        numItems = len(np.unique(self.entireFile))
        optFilter = BloomFilter(numItems, self.fp)
        return optFilter.memory()

    def getSupportSizeEst(self):
        '''
            returns support size estimation using Good Turing estimator.
        '''
        if self.estimator == "GT":
            gt = goodTuringEst(self.samplePool, self.K)
            kHat = gt.getSupportSize()
        elif self.estimator == "GT-bound":
            gt = goodTuringEst(self.samplePool, self.K)
            kHat = gt.boundSupportSize(alpha=0.05)
        elif self.estimator == "Unseen":
            # Use value we passed in (from MATLAB)
            kHat = self.unseenEst
        else:
            print(self.estimator, "not implemented")
            kHat = None

        return kHat

    def initBloomFilter(self, k, p, data):
        '''
            Initializes Bloom filter with specified number of items and a nominal FPR
        '''
        bf = BloomFilter(k, p);
        # for item in data:
        #     bf.add(item)

        return bf









