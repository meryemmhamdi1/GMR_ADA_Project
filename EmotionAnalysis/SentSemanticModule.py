# Implementation of Word level emotionalities
from __future__ import division

import numpy as np
from nltk.collocations import *
import nltk
import logging
from gensim.models import word2vec

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

    clean_pmi_dict = dict(clean_pmi)
    return clean_pmi_dict


def compute_matrix_sentences_list(transcript_words, galc_lexicon, clean_pmi_dict):
    """

    :param clean_pmi_dict:
    :param transcript_words: we can pass any version of the bag of words
    :param galc_lexicon:
    :return:
    """

    sm_list = list_galc_lexicon(galc_lexicon)
    emotions = galc_lexicon.columns.values
    matrix_sentences_list = []
    for i in range(0, len(transcript_words)): # Iterate over all sentences
        # print "Tokenized Sentence :"
        # print transcript_words[i]
        " Initialize matrix for each sentence "
        w, h = len(transcript_words[i]), 10
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for (word, tag) in transcript_words[i]: # Iterate over all words in the sentence
            j = 0
            for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
                total_pmi = 1
                for representative_word in sm_list[emotion]:
                    r = len(sm_list[emotion])
                    total_pmi *= clean_pmi_dict.get((word, representative_word), 1)
                matrix_sentence[j][k] = np.power(total_pmi,1/r)
                j += 1 # increment index of representative words
            k += 1 # increment index of transcript words
        # print matrix_sentence
        # apply_syntactic_rules(transcript_words[i], matrix_sentence)
        # append the matrix_sentence to the global list for all sentences
        matrix_sentences_list.append(matrix_sentence)
    #save_list(matrix_sentences_list, "EmotionResults/matrix_sentences_list.txt")
    return matrix_sentences_list


def train_word2Vec_model(num_features_v, min_word_count_v, num_workers_v, context_v, downsampling_v, tweets, model_name):
    # Import the built-in logging module and configure it so that Word2Vec 
    # creates nice output messages
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = num_features_v    # Word vector dimensionality                      
    min_word_count = min_word_count_v   # Minimum word count                        
    num_workers = num_workers_v       # Number of threads to run in parallel
    context = context_v          # Context window size                                                                                    
    downsampling = downsampling_v   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print ("Training model...")
    model = word2vec.Word2Vec(tweets, workers=num_workers, 
                size=num_features, min_count = min_word_count, 
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_name)
    return model
    
def compute_matrix_sentences_list_word2vec(transcript_words, galc_lexicon,model):
    """

    :param clean_pmi_dict:
    :param transcript_words: we can pass any version of the bag of words
    :param galc_lexicon:
    :return:
    """

    sm_list = list_galc_lexicon(galc_lexicon)
    emotions = galc_lexicon.columns.values
    matrix_sentences_list = []
    for i in range(0, len(transcript_words)): # Iterate over all sentences
        # print "Tokenized Sentence :"
        # print transcript_words[i]
        " Initialize matrix for each sentence "
        w, h = len(transcript_words[i]), 10
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for (word, tag) in transcript_words[i]: # Iterate over all words in the sentence
            j = 0
            for emotion in range(0, len(emotions)): # Iterate over all emotions => fill in the emotional vectors for all words
                total_similarity = 1
                for representative_word in sm_list[emotion]:
                    r = len(sm_list[emotion])
                    if word in model and representative_word in model:
                        total_similarity *= model.similarity(word, representative_word) * 10
                matrix_sentence[j][k] = np.power(total_similarity,1/r)
                j += 1 # increment index of representative words
            k += 1 # increment index of transcript words
        # print matrix_sentence
        # apply_syntactic_rules(transcript_words[i], matrix_sentence)
        # append the matrix_sentence to the global list for all sentences
        matrix_sentences_list.append(matrix_sentence)
    #save_list(matrix_sentences_list, "EmotionResults/matrix_sentences_list.txt")
    return matrix_sentences_list