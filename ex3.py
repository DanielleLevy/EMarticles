"""
Jonathan Mandl 211399175
Danielle Hodaya Shrem 208150433
"""
import math
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
K=10
num_clusters = 9
v_size = 6800
epsilon = 0.0001
max_iterations = 100
threshold = 0.0001

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

def read_topics(file_name):
    topics=[]
    with open(file_name, "r") as file:
        lines = file.readlines()
        for i in range(0,len(lines),2):
            topic=lines[i].strip()
            topics.append(topic)
    return topics


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
        correct = clustersMap[i][dominantTopic[i]]
        acc+=correct/len(clusterArts[i])
    return acc/num_clusters,clusterS,clustersMap
def init_wti(articles):
    #calc the wti based on article.probtobeincluster:
    Wti = np.zeros((len(articles), num_clusters))
    for i,article in enumerate(articles):
        Wti[i] = article.probtobeincluster
    return Wti
def m_step(articles, Wti, vocabulary, lamda):
    # articles: List of article representations
    # Wti: Responsibility matrix
    # vocabulary: List of words in the corpus
    # lambda_: Smoothing parameter for Pik
    #clac ntk based on words in each article:
    Ntk={}
    Nt = {}
    for i,article in enumerate(articles):
        Ntk[article.id-1] = article.words
        Nt[i] = article.count_words
    #clac alphas:
    alphas = []
    for i in range(num_clusters):
        alpha=sum(Wti[:,i])/float(len(articles))
        if alpha<=epsilon:
            alpha=epsilon
        alphas.append(alpha)
    #normlize the alphas:
    sumalpha=sum(alphas)
    for i in range(num_clusters):
        alphas[i]/=sumalpha
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
            Pk[word] = (numerator+lamda)/(denominator + len(vocabulary) * lamda)
        Pik[i]=Pk

    return alphas, Pik, Ntk

def e_step(alphas,pik,ntk,wti,articles):
    #prevent underflow with log:
    Zti = {}
    for t in ntk.keys():
        zi = []
        for i in range(num_clusters):
            value = math.log(alphas[i])
            for k in ntk[t].keys():
                value += ntk[t][k] * math.log(pik[i][k])
            zi.append(value)
        Zti[t] = zi
        m = max(zi)
        denominator = 0.0
        for i in range(num_clusters):
            if zi[i] - m >= -K:
                denominator += math.exp(zi[i] - m)
        for i in range(num_clusters):
            if zi[i] - m < -K:
                wti[t][i] = 0.0
            else:
                numerator = math.exp(zi[i] - m)
                wti[t][i] = numerator / denominator
        articles[t].set_probtobeincluster(wti[t])
    return wti ,Zti

def likelihood(Zti):
    lan = 0
    for t in Zti.keys():
        m = max(Zti[t])
        lan += m
        sum = 0
        for z in Zti[t]:
            if z - m >= -K:
                sum += math.exp(z - m)
        lan += math.log(sum)
    return lan

def perplexity(articles, ntk, pik, lamda, words):
    perplexity = 0
    for t in articles:
        prob = 0
        cluster=t.get_cluster()
        for k in ntk[t.id-1]:
            prob_for_cluster = (pik[cluster][k]* t.count_words+lamda)/ (t.count_words + len(words) * lamda)
            prob += math.log(prob_for_cluster)*ntk[t.id-1][k]
        perplexity += math.exp(-prob / t.count_words)
    return perplexity / len(articles)
def main(develop_file,topics_file,lamda):
    events = read_file(develop_file)
    words = filter_rare_words(events)
    articles = create_articles(develop_file, words)
    wti = init_wti(articles)
    likelihoods_per_iter = []
    perplexities_per_iter = []
    # Create lists to store results for each lambda value
    for i in range(max_iterations):
        accuracy, clusterS, clustersMap = acc(articles, words)
        # maximization step:
        alphas, Pik, ntk = m_step(articles, wti, words,lamda)
        # expectation step:
        wti, Zti = e_step(alphas, Pik, ntk, wti, articles)
        # Calculate and append current likelihood to the list
        current_likelihood = likelihood(Zti)
        likelihoods_per_iter.append(current_likelihood)
        perplexities_per_iter.append(perplexity(articles, ntk, Pik, lamda, words))
        if i > 0:
            # Calculate the relative change in likelihood
            previous_likelihood = likelihoods_per_iter[-2]
            relative_change = (current_likelihood - previous_likelihood) / abs(previous_likelihood)
            if relative_change < 0:
                # raise execption because the algorithem should increase the likelihood:
                raise ValueError("Likelihood decreased. Stopping.")

            # Check if the relative change is below the threshold
            if relative_change < threshold:
                print(f"Algorithm converged at iteration {i} with relative change {relative_change:.4f}. Stopping.")
                break

        print(f"Iteration {i}, Likelihood: {current_likelihood}, perplexity: {perplexities_per_iter[-1]}")
        # Assuming likelihoods_per_iter and perplexities_per_iter are lists containing the values over iterations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(likelihoods_per_iter, marker='o')
    plt.title('Log Likelihood over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')

    plt.subplot(1, 2, 2)
    plt.plot(perplexities_per_iter, marker='o', color='red')
    plt.title('Perplexity over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')

    plt.tight_layout()

    topics = read_topics(topics_file)

    rows = []
    for cluster in range(num_clusters):
        row = {}
        row['cluster'] = cluster + 1
        # Create a dictionary for the current row
        for topic in topics:
            row[topic] = clustersMap[cluster].get(topic, 0)
        row['size'] = clusterS[cluster]
        # Add the row dictionary to the list
        rows.append(row)

    # Create the DataFrame from the list of rows
    confusion_matrix = pd.DataFrame(rows)

    # Sort the DataFrame based on Cluster Size in descending order
    confusion_matrix = confusion_matrix.sort_values(by='size', ascending=False).reset_index(drop=True)

    # Save the DataFrame to a CSV file
    confusion_matrix.to_csv("confusion_matrix.csv")

    plt.show()


    return accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ex3.py <develop.txt> <topics.txt>")
        sys.exit(1)
    develop_file, topics_file = sys.argv[1:3]
    lamdas = [float(i) / 100 for i in range(2, 11, 2)]
    accuracies = []
    for lamda in lamdas:
        accuracy = main(develop_file, topics_file, lamda)
        accuracies.append(accuracy)
    # Find the index of the maximum accuracy
    best_accuracy_index = accuracies.index(max(accuracies))
    best_lambda = lamdas[best_accuracy_index]
    best_accuracy = accuracies[best_accuracy_index]

    # Print the best lambda value and accuracy achieved
    print(f"The best lambda value is: {best_lambda} with the best accuracy: {best_accuracy}")
    print(lamdas)
    print(accuracies)

    # Plotting the accuracies with respect to lambda values
    plt.figure(figsize=(8, 6))
    plt.plot(lamdas, accuracies, marker='o')
    plt.title('Accuracy vs. Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.xticks(lamdas)
    plt.grid(True)
    plt.tight_layout()
    plt.show()







