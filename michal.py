"""
Yosef Zumer 318808573
Michal Rahimi 316614361
"""
import math
import sys
from collections import Counter
#import matplotlib.pyplot as plt

import numpy as np

EPSILON = 0.0001
K = 10
CLUSTERS = 9


# returns dictionary of articles and their topics
def topicsDictionary(developSet):
    topics = {}
    count = 1
    g = open(developSet, "r")
    lines = g.readlines()
    for i in range(0, len(lines), 4):
        line = lines[i].split('\t')[2:]
        temp = line[-1][:-2]
        line[-1] = temp
        topics[count] = line
        count += 1
    return topics


def readFile(developSet, wordsDic):
    articles = {}  # dictionary of dictionaries- for each article we save the frequency of each word
    g = open(developSet, "r")
    lines = g.readlines()
    count = 1
    for i in range(2, len(lines), 4):
        line = lines[i].split()
        artFreq = Counter(line)  # frequency of eah word in each article
        val = {}
        for word in artFreq.keys():
            if word in wordsDic.keys():
                val[word] = artFreq[word]
        articles[count] = val
        count += 1
    return articles


# counts the occurrence of all words in all articles
def wordOccurrences(developSet):
    events = []
    g = open(developSet, "r")
    lines = g.readlines()
    for i in range(2, len(lines), 4):
        line = lines[i].split()
        events += line
    wordsOcuur = Counter(events)
    toRemove = []
    for key in wordsOcuur.keys():
        if wordsOcuur[key] <= 3:
            toRemove.append(key)
    for r in toRemove:
        wordsOcuur.pop(r, None)
    return wordsOcuur


# initialize the partition to clusters
def initialization(keys):
    w = {}
    for key in keys:
        temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        temp[(key - 1) % CLUSTERS] = 1.0
        w[key] = temp
    return w


def maximization(Wti, lamda, Ntk, words, Nt):
    alphaArray = []
    for i in range(CLUSTERS):
        sumi = 0.0
        for t in Wti.keys():
            sumi += Wti[t][i]
        val = sumi / float(len(Wti.keys()))
        if val <= EPSILON:
            val = EPSILON
        alphaArray.append(val)
    # Normalization
    sumAllAlpha = sum(alphaArray)
    for i in range(len(alphaArray)):
        normal = alphaArray[i] / sumAllAlpha
        alphaArray[i] = normal
    Pik = {}
    for i in range(CLUSTERS):
        pk = {}
        denominator = 0.0
        for t in Wti.keys():
            denominator += Wti[t][i] * Nt[t]
        for k in words:
            numerator = 0.0
            for t in Wti.keys():
                if k in Ntk[t].keys():
                    numerator += Ntk[t][k] * Wti[t][i]
            pk[k] = (numerator + lamda) / (denominator + (len(words) * lamda))
        Pik[i] = pk
    return alphaArray, Pik


def expectation(alphas, Pik, Ntk, Wti):
    Zti = {}
    for t in Wti.keys():
        zi = []
        for i in range(CLUSTERS):
            val = math.log(alphas[i])
            for k in Ntk[t].keys():
                val += Ntk[t][k] * math.log(Pik[i][k])
            zi.append(val)
        Zti[t] = zi
        m = max(zi)
        denominator = 0.0
        for j2 in range(CLUSTERS):
            if zi[j2] - m >= -K:
                denominator += math.exp(zi[j2] - m)
        for j in range(CLUSTERS):
            if zi[j] - m < -K:
                Wti[t][j] = 0.0
            else:
                numerator = math.exp(zi[j] - m)
                Wti[t][j] = numerator / denominator
    return Wti, Zti


def likelihood(Zti):
    logL = 0
    for t in Zti.keys():
        m = max(Zti[t])
        logL += m
        sumi = 0
        for z in Zti[t]:
            if z - m >= -K:
                sumi += math.exp(z - m)
        logL += math.log(sumi)
    return logL


def perplexity(wti, Ntk, Pik, lamda, Nt, words):
    perplexity = 0
    for t in wti.keys():
        probability = 0
        c = np.argmax(wti[t])
        for k in Ntk[t]:
            pX = (Pik[c][k] * Nt[t] + lamda) / (Nt[t] + len(words) * lamda)
            probability += math.log(pX) * Ntk[t][k]
        perplexity += math.exp(probability / -Nt[t])
    return perplexity / len(wti.keys())


def accuracy(Wti, topics):
    acc = 0.0
    clusterArts = {}  # dictionary of each cluster and his articles that clustered to
    for i in range(CLUSTERS):
        clusterArts[i] = []
    for t in Wti.keys():
        probs = Wti[t]
        max_value = max(probs)
        clusterArts[probs.index(max_value)].append(t)
    clustersMap = {}
    clusterS = {}
    rowsOfconfusionMat = {}
    for i in range(CLUSTERS):
        allTopicsOFArtInTheClusterI = []
        for t in clusterArts[i]:
            allTopicsOFArtInTheClusterI += topics[t]
        countA = Counter(allTopicsOFArtInTheClusterI)
        rowsOfconfusionMat[i] = countA
        dominetTopicCount = max(countA.values())  # the topic with the biggest frequency
        # each cluster is labeled as the topic with the greatest frequency
        clustersMap[i] = list(countA.keys())[list(countA.values()).index(dominetTopicCount)]
        clusterAcc = dominetTopicCount / float(len(clusterArts[i]))
        acc += clusterAcc
        clusterS[i] = len(clusterArts[i])
    return acc / CLUSTERS, clusterS, rowsOfconfusionMat


def run(develop, lamda):
    articlesToTopics = topicsDictionary(develop)
    wordsOcuur = wordOccurrences(develop)
    Ntk = readFile(develop, wordsOcuur)
    Wti = initialization(Ntk.keys())
    Nt = {}  # every article with his len
    for t in Ntk.keys():
        Nt[t] = sum(Ntk[t].values())
    acc = 0
    # initialization
    recent1, recent2 = 10**8, 0
    i = 0
    likelihoodArr = []
    perplexityArr = []
    finalCluster = {}
    rows = {}
    while abs(recent1 - recent2) > 10**3 or i < 2:
        acc, clusterDic, confusionMatRows = accuracy(Wti, articlesToTopics)
        finalCluster = clusterDic
        rows = confusionMatRows
        alphas, Pik = maximization(Wti, lamda, Ntk, wordsOcuur.keys(), Nt)
        Wti, Zti = expectation(alphas, Pik, Ntk, Wti)
        li = likelihood(Zti)
        per = perplexity(Wti, Ntk, Pik, lamda, Nt, wordsOcuur.keys())
        perplexityArr.append(per)
        likelihoodArr.append(li)
        print("Likelihood of epoch " + str(i) + ": " + str(li))
        recent1 = recent2
        recent2 = li
        i += 1
    # Confusion Matrix
    confusionMat = [['cluster index', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn', 'size']]
    sortClusterByLen = {k: v for k, v in sorted(finalCluster.items(), key=lambda item: item[1], reverse=True)}
    for c in sortClusterByLen.keys():
        newRow = [str(c)]
        for top in confusionMat[0][1:]:
            newRow.append(rows[c][top])
        newRow[9] = finalCluster[c]
        confusionMat.append(newRow)
    # Perplexity and Likelihood graphs
    print(np.array(confusionMat))
    createGraph([j for j in range(i)], likelihoodArr, 'Likelihood', 'Likelihood per iteration')
    createGraph([j for j in range(i)], perplexityArr, 'Perplexity', 'Perplexity per iteration:')
    return acc


def findBestLamda():
    lamda = [float(i)/100 for i in range(2, 11, 2)]
    res = []
    for l in lamda:
        res.append(run(develop, l))
    print(res)


def createGraph(xpoints, ypoints, ylabel, title):
    plt.plot(xpoints, ypoints)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    develop = sys.argv[1]
    topics = sys.argv[2]
    run(develop, 0.1)
    #findBestLamda()
