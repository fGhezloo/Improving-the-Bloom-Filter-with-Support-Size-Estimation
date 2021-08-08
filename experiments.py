from SyntheticData import dataDistribution
from utility import utility
import numpy as np
import matplotlib.pyplot as plt
from bloomfilter import BloomFilter
from RealData import fileData


### Question 1: As the number of sampled elements increases, how does the performance of our estimator change?
def testIncreasingN(dist, distName, estimators, K, numTrials, nValues=None,
                    shuffle=None, unseenEstimates=None):
    """Test how FP and true size change for a given estimator"""
    if nValues is None:
        # The default setting, can be overwritten
        nValues = [int(x) for x in np.linspace(0.05*K, 0.95*K, 20)]

    # These will be lists of lists, whose ith element is the list of values for the ith n-value
    actualSize = dict()
    for e in estimators:
        actualSize[e] = [[] for _ in range(len(nValues))]
    trueFP = dict()
    for e in estimators:
        trueFP[e] = [[] for _ in range(len(nValues))]
    optSize = [[] for _ in range(len(nValues))]

    for i in range(numTrials):
        # Should we get one sample per trial, or per trial/nValue combination?
        if shuffle is not None:
            allData = dist.sample(K, shuffle=shuffle)
        else:
            allData = dist.sample(K)

        for j, n in enumerate(nValues):
            for idx, e in enumerate(estimators):
                if e == "Unseen":
                    # Pass in the MATLAB value
                    # print("unseen estimate is", int(actualSizeUnseen[j,i]))
                    # print("j", j, "i", i)
                    test = utility(allData, 0.05, n, e, unseenEst=int(unseenEstimates[j, i]))
                else:
                    test = utility(allData, 0.05, n, e)
                trueFP[e][j].append(test.getTrueFP())
                actualSize[e][j].append(test.getSize())
                if idx == 0:
                    optSize[j].append(test.getOptSize())

    # Sort the sub-lists (the replicates for each value of n), so we can generate confidence intervals
    for e in estimators:
        actualSize[e] = [sorted(x) for x in actualSize[e]]
        trueFP[e] = [sorted(x) for x in trueFP[e]]
    optSize = [sorted(x) for x in optSize]

    CI95idx = int(numTrials*0.95)
    CI05idx = int(numTrials*0.05)
    for e in estimators:
        plt.plot(nValues, [np.mean(x) for x in trueFP[e]], label=e)
        plt.fill_between(nValues, [x[CI05idx] for x in trueFP[e]], [x[CI95idx] for x in trueFP[e]],
                         alpha=0.1)
    plt.plot(nValues, [0.05]*len(nValues), label="Nominal FP", color='k', linestyle='--')
    if numTrials > 1:
        plt.title("Mean and 90% Confidence Intervals for false positive rate\nwith " + distName + ", " +
                  str(numTrials) + " trials")
    else:
        plt.title("False positive rate with "+distName)
    plt.xlabel("Number of samples used to estimate support, n (where K="+str(K)+")")
    plt.ylabel("False positive rate")
    plt.legend()
    plt.tight_layout(pad=2)
    plt.savefig("testIncreasingN_"+distName+"_FP.png")
    plt.clf()

    for e in estimators:
        plt.plot(nValues, [np.mean(x) for x in actualSize[e]], label=e)
        plt.fill_between(nValues, [x[CI05idx] for x in actualSize[e]], [x[CI95idx] for x in actualSize[e]],
                         alpha=0.1)
    plt.plot(nValues, [np.mean(x) for x in optSize], label="optimal size", color='k', linestyle='--')
    plt.fill_between(nValues, [x[CI05idx] for x in optSize], [x[CI95idx] for x in optSize],
                     alpha=0.1)
    plt.legend()
    if numTrials > 1:
        plt.title("Mean and 90% Confidence Intervals for bloom filter size\nwith " + distName + ", "
                  + str(numTrials)+" trials")
    else:
        plt.title("Bloom filter size with " + distName)
    plt.xlabel("Number of samples used to estimate support, n (where K="+str(K)+")")
    plt.ylabel("Bloom filter size (Bits)")
    plt.ylim(0, 3 * max(optSize[-1]))
    plt.savefig("testIncreasingN_"+distName+"_size.png")
    plt.tight_layout(pad=2)
    plt.clf()


### Example usage for Question 1

# # The Zipf(1.5) distribution
# zipfDist = dataDistribution(2, [1.5])
# testIncreasingN(dist=zipfDist, distName="Zipf(1.5)", K=10000, estimator="GT", numTrials=100)
#
# # Few items (uniform on like a hundred items)
# uniformDense = dataDistribution(1, [100])
# testIncreasingN(dist=uniformDense, distName="Uniform(100)", K=10000, estimator="GT", numTrials=100)
#
# # Many items (uniform on like a thousand items)
# uniformSparse = dataDistribution(1, [1000])
# testIncreasingN(dist=uniformSparse, distName="Uniform(1000)", K=10000, estimator="GT", numTrials=100)

# DBLP institutions
institutions = fileData("institutions.txt", verbose=True)  #3752 unique institutions
print("institutions")
print([int(x) for x in np.linspace(0.05*959292, 0.95*959292, 20)])

# Get experimental data for dblp on unseen estimator
dblp_exp1 = [2.6221,    3.0095,    3.8867,   3.4678,    3.2556,    3.3570,    3.2909,    3.3813,    3.2062,    3.1554,    4.0591,    4.4277,    4.4762,    4.3789,    4.6350,    4.6261,    4.8147,    4.6982,    4.7104,   4.5762]
dblp_exp1 = [[1e3*x] for x in dblp_exp1]
#
# testIncreasingN(dist=institutions, distName="DBLP-institutions",
#                 K=959292, estimators=["GT", "GT-bound", "Unseen"], numTrials=1,
#                 unseenEstimates=np.array(dblp_exp1))

dblp_Exp2 = np.loadtxt("institution_ex2.txt")
dblp_Exp2 = dblp_Exp2.reshape(40, -1)
print(dblp_Exp2.shape)
testIncreasingN(dist=institutions, distName="DBLP-institutions-shuffled",
                K=959292, estimators=["GT", "GT-bound", "Unseen"], numTrials=40, shuffle=True,
                unseenEstimates=dblp_Exp2.T)

# Hamlet
hamlet = fileData("hamletWords.txt", verbose=True)  # 5084 unique words
print("hamlet")
print([int(x) for x in np.linspace(0.05*30780, 0.95*30780, 20)])

# hamlet_exp1 = [5.5506,    3.7347,    5.6528,   6.4998,    6.1890,   6.0985,   6.0918,  6.4367,   6.5590,    6.6625,    6.8051,    6.9606,    7.2218,    7.0709,    7.5288,    7.1900,    6.6752,    6.7734,   4.3898,    5.5833]
# hamlet_exp1 = [[1e3*x] for x in hamlet_exp1]
# testIncreasingN(dist=hamlet, distName="hamlet",
#                 K=30780, estimators=["GT", "GT-bound", "Unseen"], numTrials=1,
#                 unseenEstimates=np.array(hamlet_exp1))
#
# hamlet_Exp2 = np.loadtxt("hamlet_ex2.txt")
# hamlet_Exp2 = hamlet_Exp2.reshape(40, -1)
# testIncreasingN(dist=hamlet, distName="hamlet-shuffled",
#                 K=30780, estimators=["GT", "GT-bound", "Unseen"],
#                 numTrials=40, shuffle=True, unseenEstimates=hamlet_Exp2.T)


### Question 2: For a specific choice of n (relative to K?), how does the performance of the method
### change as K increases?
def testIncreasingK(dist, distName, estimators, nFunctionName, nFunctionDesc,
                    nFunction, numTrials, Ks=None, shuffle=None, unseenEstimates=None):
    """nFunction will be applied as n = nFunction(K) where K is the total file size"""
    if Ks is None:
        Ks = [int(x) for x in np.logspace(3, 6, 8)]

    # These will be lists of lists, whose ith element is the list of values for the ith n-value
    actualSize = dict()
    for e in estimators:
        actualSize[e] = [[] for _ in range(len(Ks))]
    trueFP = dict()
    for e in estimators:
        trueFP[e] = [[] for _ in range(len(Ks))]
    optSize = [[] for _ in range(len(Ks))]

    for i in range(numTrials):
        for j, K in enumerate(Ks):
            K = int(K)
            if shuffle is not None:
                allData = dist.sample(K, shuffle=shuffle)
            else:
                allData = dist.sample(K)
            n = nFunction(K)

            for idx, e in enumerate(estimators):
                if e == "Unseen":
                    test = utility(allData, 0.05, n, e, unseenEst=int(unseenEstimates[j,i]))
                else:
                    test = utility(allData, 0.05, n, e)
                trueFP[e][j].append(test.getTrueFP())
                actualSize[e][j].append(test.getSize())
                if idx == 0:
                    optSize[j].append(test.getOptSize())

    # Sort the sub-lists (the replicates for each value of n), so we can generate confidence intervals
    for e in estimators:
        actualSize[e] = [sorted(x) for x in actualSize[e]]
        trueFP[e] = [sorted(x) for x in trueFP[e]]
    optSize = [sorted(x) for x in optSize]

    CI95idx = int(numTrials * 0.95)
    CI05idx = int(numTrials * 0.05)

    startingKIdx = 10 # the first few are just too noisy

    for e in estimators:
        plt.plot(Ks[startingKIdx:-1], [np.mean(x) for x in trueFP[e]][startingKIdx:-1], label=e)
        plt.fill_between(Ks[startingKIdx:-1], [x[CI05idx] for x in trueFP[e]][startingKIdx:-1],
                         [x[CI95idx] for x in trueFP[e]][startingKIdx:-1],
                         alpha=0.1)
    plt.plot(Ks[startingKIdx:-1], [0.05] * len(Ks[startingKIdx:-1]), label="Nominal FP",
             color='k', linestyle='--')
    if numTrials > 1:
        plt.title("Mean and 90% Confidence Intervals for false positive rate\nwith " + distName +
                  ",\n"+nFunctionDesc+"\n" + str(numTrials) + " trials")
    else:
        plt.title("False positive rate with " + distName +
                  ",\n" + nFunctionDesc)
    plt.xlabel("Total number of items in file, K")
    plt.ylabel("False positive rate")
    plt.legend()
    plt.tight_layout(pad=2)
    plt.savefig("testIncreasingK_" + distName + "_" + nFunctionName + "_FP.png")
    plt.clf()

    for e in estimators:
        plt.plot(Ks[startingKIdx:-1], [np.mean(x) for x in actualSize[e]][startingKIdx:-1], label=e)
        plt.fill_between(Ks[startingKIdx:-1], [x[CI05idx] for x in actualSize[e]][startingKIdx:-1],
                         [x[CI95idx] for x in actualSize[e]][startingKIdx:-1],
                         alpha=0.1)
    plt.plot(Ks[startingKIdx:-1], [np.mean(x) for x in optSize][startingKIdx:-1], label="optimal size",
             color='k', linestyle='--')
    plt.fill_between(Ks[startingKIdx:-1], [x[CI05idx] for x in optSize][startingKIdx:-1],
                     [x[CI95idx] for x in optSize][startingKIdx:-1],
                     alpha=0.1)
    plt.legend()
    if numTrials > 1:
        plt.title("Mean and 90% Confidence Intervals for bloom filter size\nwith " + distName +
                  ",\n"+nFunctionDesc+"\n" + str(numTrials) + " trials")
    else:
        plt.title("Bloom filter size with " + distName + ",\n"+nFunctionDesc)
    plt.xlabel("Total number of items in file, K")
    plt.ylabel("Bloom filter size (Bits)")
    plt.ylim(0, 3*max(optSize[-1]))
    plt.tight_layout(pad=2)
    plt.savefig("testIncreasingK_" + distName + "_" + nFunctionName + "_size.png")
    plt.clf()


# The Zipf(1.5) distribution
# zipfDist = dataDistribution(2, [1.5])
# testIncreasingK(dist=zipfDist, distName="Zipf(1.5)", nFunction=lambda k : int(0.1*k),
#                 nFunctionName="TenPercent", nFunctionDesc="n=0.1K",
#                 estimator="GT", numTrials=100)
#
# testIncreasingK(dist=zipfDist, distName="Zipf(1.5)", nFunction=lambda k : int(k/(2*np.log(k))),
#                 nFunctionName="klogk", nFunctionDesc="n=K/2logK",
#                 estimator="GT", numTrials=100)

def optSizeKItems(k):
    """What's the optimal size of a bloom filter with k distinct items?"""
    bf = BloomFilter(k, 0.05)
    return int(bf.memory()/80)

# testIncreasingK(dist=zipfDist, distName="Zipf(1.5)", nFunction=optSizeKItems,
#                 nFunctionName="optSizeK", nFunctionDesc="n is the optimal size of a K-item filter",
#                 estimator="GT", numTrials=100)

print("DBLP")
print([optSizeKItems(int(k)) for k in np.logspace(1, np.log10(959292), num=20)])
print("hamlet")
print([optSizeKItems(int(k)) for k in np.logspace(1, np.log10(30780), num=20)])

# Real data
exp5_dblp = [0.0001,    0.0001,    0.0004,    0.0008,    0.0009,    0.6256,    0.0012,    1.4380,    1.0319,    1.1338,    0.0281,    0.0078,    0.2027,    0.0182,    0.1187,    0.0259,    0.0269,  0.0281,    0.0330,    0.0349]
exp5_dblp = [[1e5*x] for x in exp5_dblp]
# testIncreasingK(dist=institutions, distName="DBLP-institutions",
#                 Ks=np.logspace(1, np.log10(959292), num=20),
#                 nFunction=optSizeKItems, nFunctionName="optSizeK",
#                 nFunctionDesc="n is the optimal size of a K-item filter",
#                 estimators=["GT", "GT-bound", "Unseen"], numTrials=1, shuffle=False,
#                 unseenEstimates=np.array(exp5_dblp))


exp5_ham = [2.5000,    2.5000,    2.5000,    2.5000,    0.0056,    0.8161,    0.0071,    0.0135,    0.0433,    0.0827,    1.0403,    0.0693,    0.1593,    0.1561,    0.4866,    0.6603,    0.3209,    0.5047,    0.6464,    0.6107]
exp5_ham = [[1e4*x] for x in exp5_ham]
# testIncreasingK(dist=institutions, distName="hamlet",
#                 Ks=np.logspace(1, np.log10(30780), num=20),
#                 nFunction=optSizeKItems, nFunctionName="optSizeK",
#                 nFunctionDesc="n is the optimal size of a K-item filter",
#                 estimators=["GT", "GT-bound", "Unseen"], numTrials=1, shuffle=False,
#                 unseenEstimates=np.array(exp5_ham))


# ham_exp6 = np.loadtxt("hamlet_ex6.txt")
# ham_exp6 = ham_exp6.reshape(40, -1)
# testIncreasingK(dist=institutions, distName="hamlet-shuffled",
#                 Ks=np.logspace(1, np.log10(30780), num=20),
#                 nFunction=optSizeKItems, nFunctionName="optSizeK",
#                 nFunctionDesc="n is the optimal size of a K-item filter",
#                 estimators=["GT", "GT-bound", "Unseen"], numTrials=40, shuffle=True,
#                 unseenEstimates=ham_exp6.T)

# dblp_exp6 = np.loadtxt("institution_ex6.txt")
# dblp_exp6 = dblp_exp6.reshape(40, -1)
# testIncreasingK(dist=institutions, distName="DBLP-institutions shuffled",
#                 Ks=np.logspace(1, np.log10(959292), num=20),
#                 nFunction=optSizeKItems, nFunctionName="optSizeK",
#                 nFunctionDesc="n is the optimal size of a K-item filter",
#                 estimators=["GT", "GT-bound", "Unseen"], numTrials=40, shuffle=True,
#                 unseenEstimates=dblp_exp6.T)
