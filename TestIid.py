# Does it appear that our data might be coming iid from some distribution?
# Plot the appearance of new things vs. bootstrapped iid samples

import numpy as np
import re
import matplotlib.pyplot as plt

def iidPlots(fileName, numReps, fileDesc):
    # Read file
    f = open(fileName, "r")
    data = f.read()
    dataList = re.split("\n", data)

    # Get the plot for this data
    fracObs = fracSeen(dataList)

    # Get bootstrap replicates
    replicates = np.zeros(shape=(len(fracObs), numReps))
    for idx in range(numReps):
        # Shuffle the data
        shuffledData = np.random.choice(a=dataList,
                                        size=len(dataList),
                                        replace=True)
        rep = fracSeen(shuffledData)
        replicates[:,idx] = rep


    CI95idx = int(numReps*0.95)
    CI05idx = int(numReps*0.05)
    plt.plot(np.arange(len(fracObs)), fracObs, label="Observed")
    plt.fill_between(np.arange(len(fracObs)),
                     [sorted(replicates[i, :])[CI05idx] for i in range(len(fracObs))],
                     [sorted(replicates[i, :])[CI95idx] for i in range(len(fracObs))],
                     alpha=0.2, label="Bootstrapped 90% CI for IID data")
    plt.ylabel("Cumulative fraction of items observed")
    plt.xlabel("Number of items observed")
    plt.title("Test for iid property of " + fileDesc + " data")
    plt.legend()
    plt.tight_layout(pad=2)

    plt.savefig("testIId_"+fileDesc+".png")
    plt.clf()


def fracSeen(data):
    """As we read through the data, plot the cumulative fraction seen"""
    itemSet = set()
    numSeen = []
    for x in data:
        itemSet.add(x)
        numSeen.append(len(itemSet))

    fracSeen = [x/numSeen[-1] for x in numSeen]
    return fracSeen


iidPlots("hamletWords.txt", numReps=100, fileDesc="Hamlet")

iidPlots("institutions.txt", numReps=10, fileDesc="DBLP-institutions")