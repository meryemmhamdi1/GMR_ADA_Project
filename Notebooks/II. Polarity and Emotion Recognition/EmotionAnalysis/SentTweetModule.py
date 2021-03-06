from __future__ import division
import math
from tqdm import tqdm

# Implementation of Tweet Level Emotionalities
# At this level, we already have emotional vectors for each word

def make_flat_list(data_list):
    """

    :param data_list:
    :return:
    """
    flatten_list = [word for sublist in data_list for (word, tag) in sublist]
    return flatten_list

def compute_sentence_emotion_vectors(matrix_sentences_list):
    emotion_vector_list = []
    for i in range(0, len(matrix_sentences_list)):
        sum_sentence = []
        ids = [0,1,2,3,4,7,8,9]
        for j in ids: # for each emotion
            sum_words = 0
            for k in range(0, len(matrix_sentences_list[i])):
                sum_words += int(matrix_sentences_list[i][k][j])
            r = len(matrix_sentences_list[i])
            if r != 0 :
                sum_words = sum_words / r # Arithmetic mean
            sum_sentence.append(sum_words)
        emotion_vector_list.append(sum_sentence)
    return emotion_vector_list

def compute_sentence_sentiment_vectors(matrix_sentences_list):
    emotion_vector_list = []
    for i in range(0, len(matrix_sentences_list)):
        sum_sentence = []
        ids = [5,6]
        for j in ids: # for each emotion
            sum_words = 0
            for k in range(0, len(matrix_sentences_list[i])):
                sum_words += int(matrix_sentences_list[i][k][j])
            r = len(matrix_sentences_list[i])
            if r != 0 : 
                sum_words = sum_words / r
            sum_sentence.append(sum_words)
        emotion_vector_list.append(sum_sentence)
    return emotion_vector_list

def compute_emotionalities(sentence_vectors):
    emotionalities = []
    threshold = 0 # THRESHOLD PARAMETER TO BE FINE TUNED (0 for lexicon, 0.2 for pmi)
    for i in range(0,len(sentence_vectors)):
        sentence_vector = sentence_vectors[i]
        mylist = [0 if math.isnan(x) else x for x in sentence_vector]
        if (max(mylist) > threshold): #Threshold 
            emotionalities.append(sentence_vectors[i].index(max(mylist)))
        else: 
            emotionalities.append(8)
    return emotionalities

def compute_sentiments(sentence_vectors_sent,emotionalities):
    sentiments = []
    threshold = 0 # THRESHOLD PARAMETER TO BE FINE TUNED (0 for lexicon, 0.2 for pmi)
    for i in range(0,len(sentence_vectors_sent)):
        sentence_vector = sentence_vectors_sent[i]
        mylist = [0 if math.isnan(x) else x for x in sentence_vector]
        if (max(mylist) > threshold): #Threshold 
            sentiments.append(sentence_vectors_sent[i].index(max(mylist)))
        else:
            # To increase Recall, we also use emotionalities, in case a tweet is neutral
            if emotionalities[i] in [0,2,3,5]:
                sentiments.append(0) # Negative Emotion
            if emotionalities[i] in [1,4,6,7]:
                sentiments.append(1) # Positive Emotion
            if emotionalities[i] == 8:
                sentiments.append(2) # Otherwise, we just return Neutral
    return sentiments