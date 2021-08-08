# Convenience file for reading data in a way that interfaces
# with our experiments

import re
import numpy as np

class fileData(object):

    def __init__(self, fileName, verbose=True):
        f = open(fileName, "r")
        data = f.read()
        self.dataList = re.split("\n", data)
        if verbose:
            print(fileName)
            print(len(np.unique(self.dataList)), "unique,")
            print(len(self.dataList), "total")

    def sample(self, K, shuffle=False):
        """Return the first K words from the file.
        If shuffle is true, we return a random K words; otherwise,
        return the first K"""
        if shuffle:
            return np.random.choice(self.dataList, replace=False, size=K)
        else:
            return self.dataList[0:K]
