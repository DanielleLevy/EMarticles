"""
Jonathan Mandl 211399175
Danielle Hodaya Shrem 208150433
"""
import math
import sys
from collections import Counter
#import matplotlib.pyplot as plt
import numpy as np
K=10
num_clusters = 9
v_size = 6800
epsilon = 0.0001
class article:
    def __init__(self, id,lines, words,topic):
        self.id = id
        self.cluster = id%9
        self.lines = lines
        self.words = words
        self.topic = topic
        self.count_words = sum(words.values())
        self.probtobeincluster = np.zeros(num_clusters)
        self.probtobeincluster[self.cluster] = 1

    def get_cluster(self):
        self.cluster = np.argmax(self.probtobeincluster)
        return self.cluster
    def set_probtobeincluster(self,probs):
        self.probtobeincluster = probs
#return a list of words=events:
def read_file(file_name):
    events = []
    with open(file_name, "r") as file:
        lines = file.readlines()
        for i in range(2, len(lines),4):
            line=lines[i].split()
            events = events+line[:-1]
    return events

def filter_rare_words(events):
    counter = Counter(events)
    #return a dict of words and their count:
    return {word: count for word, count in counter.items() if count >= 4}

#function that will go over the develop and create a list of articles:
def create_articles(develop_file,wordsdict):
    counter=1
    articles = []
    with open(develop_file, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            #extarct the topic:
            lineTopic = lines[i].split('\t')[2:]
            temp = lineTopic[-1][:-2]
            lineTopic[-1] = temp
            id = counter
            counter+=1
            line=lines[i+2]
            words = lines[i+2].split()
            counterWords = Counter(words)
            #check if the word is in the wordsdict:
            words = {word: count for word, count in counterWords.items() if word in wordsdict}
            articles.append(article(id, line, words, lineTopic))
    return articles




def initialize_alphas(num_clusters):
    return np.full(num_clusters, 1 / num_clusters)  # Equally distribute the initial cluster probabilities



def acc(articles,words):
    #calc which article in each cluster:
    clusterArts = {}
    for i in range(num_clusters):
        clusterArts[i] = []
    for article in articles:
        clusterArts[article.get_cluster()].append(article.id)
    #calc clusterS(how many articles in each cluster based on clusterArts):
    clustersMap = {}
    clusterS = {}
    for i in range(num_clusters):
        clustersMap[i] = {}
        clusterS[i] = len(clusterArts[i])
    for article in articles:
        cluster = article.get_cluster()
        for topic in article.topic:
            if topic in clustersMap[cluster]:
                clustersMap[cluster][topic] += 1
            else:
                clustersMap[cluster][topic] = 1
    dominantTopic = {}
    for i in range(num_clusters):
        dominantTopic[i] = max(clustersMap[i], key=clustersMap[i].get)
    #calc the accuracy:
    correct = 0
    acc=0
    for i in range(num_clusters):
        correct += clustersMap[i][dominantTopic[i]]
        acc+=correct/len(clusterArts[i])
    return acc/num_clusters,clusterS,clustersMap
def init_wti(articles):
    #calc the wti based on article.probtobeincluster:
    Wti = np.zeros((len(articles), num_clusters))
    for i,article in enumerate(articles):
        Wti[i] = article.probtobeincluster
    return Wti
def m_step(articles, Wti, vocabulary, lambda_=0.1):
    # articles: List of article representations
    # Wti: Responsibility matrix
    # vocabulary: List of words in the corpus
    # lambda_: Smoothing parameter for Pik
    #clac ntk based on words in each article:
    Ntk={}
    Nt = {}
    for i,article in enumerate(articles):
        Ntk[article.id] = article.words
        Nt[i] = article.count_words
    #clac alphas:
    alphas = []
    for i in range(num_clusters):
        alpha=sum(Wti[:,i])/float(len(articles))
        if alpha<=epsilon:
            alpha=epsilon
        alphas.append(alpha)
    #normlize the alphas:
    sum=sum(alphas)
    for i in range(num_clusters):
        alphas[i]/=sum
    #clac Pik:
    Pik={}
    for i in range(num_clusters):
        #clac pk=wti*ntk/nt*wti:
        Pk={}
        denominator = 0.0
        for t in range(len(Wti)):
            denominator += Wti[t][i]*Nt[t]
        for word in vocabulary:
            numerator = 0.0
            for t in range(len(Wti)):
                if word in Ntk[t]:
                    numerator += Ntk[t][word] * Wti[t][i]
            Pk[word] = (numerator+lambda_)/(denominator + len(vocabulary) * lambda_)
        Pik[i]=Pk

        return alphas, Pik





if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ex3.py <develop.txt> <topics.txt>")
        sys.exit(1)
    develop_file, topics_file = sys.argv[1:3]
    events=read_file(develop_file)
    words = filter_rare_words(events)
    articles=create_articles(develop_file,words)
    wti=init_wti(articles)
    for i in range(10):
        accuracy,clusterS,clustersMap = acc(articles,words)
        #maximization step:
        alphas, Pik = m_step(articles, wti, words)

