import tqdm
import pandas as pd

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in tqdm.tqdm(documents, desc='Computing IDF documents'):
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in tqdm.tqdm(idfDict.items(), desc='Computing IDF words'):
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def calc_tfidf(df):
    numOfWordsAll: dict = df.to_dict("index")
    bagOfWords: list = df.columns.to_list()
    tf_all: dict = {}
    tfidf_all: dict = {}

    for id, numofwords in tqdm.tqdm(numOfWordsAll.items(), desc="Computing TF"):
        tf_all.update({id: computeTF(numofwords, bagOfWords)})

    idfs = computeIDF(list(tf_all.values()))

    for id, tf in tqdm.tqdm(tf_all.items(), desc="Computing TFIDF"):
        tfidf_all.update({id: computeTFIDF(tf, idfs)})

    return pd.DataFrame(list(tfidf_all.values()), index=list(tfidf_all.keys()))

