# Implementation of Word level emotionalities
from __future__ import division
import sys
#sys.path.insert(0, "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis")
import numpy as np
from nltk.collocations import *
from gensim.models import word2vec
from EmotionAnalysis.DataPreProcessing import *
from tqdm import tqdm
from joblib import Parallel, delayed
from math import sqrt

def calculate_pmi(flatten_list_nava, unique_lexicon):
    finder = BigramCollocationFinder.from_words(flatten_list_nava, window_size=10)
    finder1 = BigramCollocationFinder.from_words(flatten_list_nava, window_size=10)

    " Without Filter: use pmi matrix"
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    pmi = finder.score_ngrams(bigram_measures.pmi)

    " With Filter: use pmi1 matrix"
    finder1.apply_freq_filter(13)
    pmi1 = finder1.score_ngrams(bigram_measures.pmi)

    " Clean Dictionary: keeping only values for which the second the word is part of galc lexicon"
    clean_pmi = []
    for ((w1, w2), value) in pmi:
        if w2 in unique_lexicon:
            clean_pmi.append(((w1, w2), value))
        if w1 in unique_lexicon:
            clean_pmi.append(((w2, w1), value))

    clean_pmi_dict = dict(clean_pmi)
    return clean_pmi_dict

def update_matrix_sentence_whole (lexicon, nava_tweets_i,matrix_sentence_whole):
    w, h = len(nava_tweets_i),10
    matrix_sentence = [[0 for x in range(w)] for y in range(h)]
    j = 0
    for word in nava_tweets_i: # for each word
        #print nava_tweets_i
        # Looking for match between that keyword and representative word in each emotion category in the lexicon
        for e in range(0,lexicon.shape[1]):
            if word in list(lexicon[str(e)]):
                matrix_sentence[e][j] = 1
        j += 1
    matrix_sentence_whole.append(matrix_sentence)
    return matrix_sentence_whole

def compute_matrix_sentences_list_lexicon(nava_tweets, lexicon):
    matrix_sentence_whole = []
    matrix_sentence_whole = Parallel(n_jobs=1)( delayed(update_matrix_sentence_whole) (lexicon,nava_tweets[i],matrix_sentence_whole) for i in tqdm(range(0,len(nava_tweets))) )
    #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    #list_ = [[[0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 0]]]
    #print matrix_sentence_whole
    return matrix_sentence_whole
    
def compute_matrix_sentences_list(tweet_words, nrc_lexicon, clean_pmi_dict):
    """

    :param clean_pmi_dict:
    :param tweet_words: we can pass any version of the bag of words
    :param nrc_lexicon:
    :return:
    """

    sm_list = list_nrc_lexicon(nrc_lexicon)
    emotions = nrc_lexicon.columns.values
    matrix_sentences_list = []
    for i in tqdm(range(0, len(tweet_words))): # Iterate over all sentences
        " Initialize matrix for each sentence "
        w, h = len(tweet_words[i]), 10
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for (word, tag) in tweet_words[i]: # Iterate over all words in the sentence
            j = 0
            for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
                total_pmi = 1
                for representative_word in sm_list[emotion]:
                    r = len(sm_list[emotion])
                    total_pmi += clean_pmi_dict.get((word, representative_word), 0)
                if word in sm_list[emotion]:
                    matrix_sentence[j][k] += 10
                else:
                    matrix_sentence[j][k] += total_pmi / r
                j += 1 # increment index of representative words
            k += 1 # increment index of tweet words
        # append the matrix_sentence to the global list for all sentences
        matrix_sentences_list.append(matrix_sentence)
    return matrix_sentences_list

    
def compute_matrix_sentences_list_word2vec(transcript_words, nrc_lexicon,model):
    """

    :param word2vec model:
    :param transcript_words: we can pass any version of the bag of words
    :param nrc_lexicon:
    :return:
    """

    sm_list = list_nrc_lexicon(nrc_lexicon)
    emotions = nrc_lexicon.columns.values
    matrix_sentences_list = []
    for i in tqdm(range(0, len(transcript_words))): # Iterate over all sentences
        " Initialize matrix for each sentence "
        w, h = len(transcript_words[i]), 10
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for (word, tag) in transcript_words[i]: # Iterate over all words in the sentence
            j = 0
            for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
                total_similarity = 0
                for representative_word in sm_list[emotion]:
                    r = len(sm_list[emotion])
                    if word in model and representative_word in model:
                        total_similarity += model.similarity(word, representative_word)
                if word in sm_list[emotion]:
                    matrix_sentence[j][k] += 10
                else:
                    matrix_sentence[j][k] += total_similarity / r  # np.power(total_similarity,1/r)
                j += 1 # increment index of representative words
            k += 1 # increment index of transcript words
        # append the matrix_sentence to the global list for all sentences
        matrix_sentences_list.append(matrix_sentence)
    return matrix_sentences_list
