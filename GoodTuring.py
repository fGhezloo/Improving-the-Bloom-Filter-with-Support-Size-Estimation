# Implement the Good-Turing estimator
# Example usage
# from GoodTuring import goodTuringEst
# words = ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'd', 'e', 'f', 'f']
# gt = goodTuringEst(words, K=20)
# gt.getSupportSize()                 # Our plug-in estimate
# gt.boundSupportSize(alpha=0.05)     # A high-probability bound

import numpy as np

class goodTuringEst(object):
    '''Class for Good-Turing missing mass estimation,
    which we will turn into a missing species estimator'''

    def __init__(self, items, K):
        '''Takes a list of items that have been seen,
        and a total number of items in the stream K.
        Builds a histogram of the number of times each thing has been seen'''
        self.K = K
        _, self.counts = np.unique(items, return_counts=True)

    def numItemsSeen(self):
        '''Total number of items (not necessarily distinct) seen'''
        return sum(self.counts)

    def getMissingMass(self):
        '''Returns a missing mass estimate - the probability mass
        of the distribution associated with yet-unseen objects'''
        numSeenOnce = sum(self.counts == 1)
        probSeenOnce = numSeenOnce/self.numItemsSeen()
        return probSeenOnce

    def boundMissingMass(self, alpha=0.05):
        '''With probability at least 1-alpha, the returned value
        does not exceed the true missing mass'''
        hoeffdingBound = np.sqrt(np.log(1.0/alpha)/(2*self.numItemsSeen()))
        return self.getMissingMass() + hoeffdingBound

    def getMissingSpecies(self):
        '''Since there are only K items total, we know no species has
        probability of being seen less than 1/K. Use this to bound the number
        of missing species. This uses the GT estimator, not the adjusted bound'''
        return int(np.ceil(self.getMissingMass()*self.K))

    def boundMissingSpecies(self, alpha=0.05):
        '''Use the upper bound on the missing mass, and the fact that no species
        has probability less than 1/K, to upper bound the number of species'''
        return int(np.ceil(self.boundMissingMass(alpha)*self.K))

    def getSupportSize(self):
        '''Return a support size estimate: seen species + missing species'''
        return len(self.counts) + self.getMissingSpecies()

    def boundSupportSize(self, alpha=0.05):
        '''With probability at least 1-alpha, the returned support size
        does not exceed the truth. Note it might be greater than the
        stream length; the user is in charge of min-ning this with the stream length'''
        return len(self.counts) + self.boundMissingSpecies(alpha)