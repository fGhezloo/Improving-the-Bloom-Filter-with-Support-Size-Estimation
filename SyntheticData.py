# Utitility functions for generating synthetic data
# We are interested in several types of synthetic distributions
# These will be thin wrappers on the Scipy frozen distribution concept

import scipy.stats


class dataDistribution(object):

    def __init__(self, distID, paramList, verbose=True):
        '''Idea: we will have a list of distributions we are
        interested in, and this will choose from that list.
        Takes a list of parameters, which you are responsible for
        getting right (sorry)'''

        if distID == 1:
            '''Distribution 1 is the uniform distribution over
            p[0] elements'''
            numElements = paramList[0]
            if verbose:
                print("Uniform distribution over", numElements, "elements")
            xVals = [i for i in range(numElements)]
            probs = [1.0/numElements]*numElements
            self.rv = scipy.stats.rv_discrete(values=(xVals, probs))

        elif distID == 2:
            '''Distribution 2 is the Zipf function, a heavy-tailed
            distribution that models natural language'''
            alpha = paramList[0]
            if verbose:
                print("Zipf distribution with shape parameter", alpha)
            self.rv = scipy.stats.zipf(alpha)


    def sample(self, numSamples):
        ''' Returns a list corresponding to numSamples iid samples
        from the distribution'''
        return self.rv.rvs(size=numSamples)

    def write(self, fileName, K):
        samples = self.sample(K)
        file = open(fileName, "w")
        for x in samples:
            file.write(str(x))
            file.write("\n")
        file.close()

