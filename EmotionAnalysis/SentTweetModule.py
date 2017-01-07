from __future__ import division

# Implementation of Tweet Level Emotionalities
# At this level, we already have emotional vectors for each word

def emotional_vectors(tweets_nava, word_emotional_vectors_dict):
    matrix_sentences_list = []
    for i in range(0, len(tweets_nava)): # iterate over all tweets
        matrix_tweet = []
        for (word, tag) in tweets_nava[i]:
            matrix_tweet.append(word_emotional_vectors_dict[word]) # search for the emotional vector of the word
        matrix_sentences_list.append(average_tweet(matrix_tweet))
    return tweet_emotion_matrix


def average_tweet(matrix_tweet):
    tweet_emotional_vector = []
    for e in range(0, 10):
        sum_ = 0
        for w in range(0, len(matrix_tweet)):
            sum_ += matrix_tweet[w][e]
        tweet_emotional_vector.append(sum_/len(matrix_tweet))
    return tweet_emotional_vector

def dominant_emotion(tweet_emotion_matrix, thresh, emotion_dict): # TODO: Which threshold is right here: should try
    dominant_emotion_tweets = []
    for i in range(0, len(tweet_emotion_matrix)):
        if max(tweet_emotion_matrix[i]) > thresh:
            dominant_emotion_tweets.append(tweet_emotion_matrix[i].index(max(tweet_emotion_matrix[i])))
        else:
            # Neutral emotion = index 11
            dominant_emotion_tweets.append(11)
        return emotion_dict.columns.values[dominant_emotion_tweets]

def make_flat_list(data_list):
    """

    :param data_list:
    :return:
    """
    flatten_list = [word for sublist in data_list for (word, tag) in sublist]
    ##save_list(flatten_list, "MediumData/flatten_list_nava.txt")
    return flatten_list

def compute_sentence_emotion_vectors(matrix_sentences_list):
    emotion_vector_list = []
    for i in range(0, len(matrix_sentences_list)):
        sum_sentence = []
        ids = [0,1,2,3,4,7,8,9]
        for j in ids: # for each emotion
            sum_words = 0
            for k in range(0, len(matrix_sentences_list[i][j])):
                sum_words += matrix_sentences_list[i][j][k]
            r = len(matrix_sentences_list[i][j])
            if r != 0 : 
                #sum_words = np.power(sum_words,1/r) # Geometric mean 
                sum_words = sum_words / r
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
            for k in range(0, len(matrix_sentences_list[i][j])):
                sum_words += matrix_sentences_list[i][j][k]
            r = len(matrix_sentences_list[i][j])
            if r != 0 : 
                #sum_words = np.power(sum_words,1/r) # Geometric mean 
                sum_words = sum_words / r
            sum_sentence.append(sum_words)
        emotion_vector_list.append(sum_sentence)
    return emotion_vector_list

def compute_emotionalities(sentence_vectors):
    emotionalities = []
    threshold = 1.0
    for i in range(0,len(sentence_vectors)):
        sentence_vector = sentence_vectors[i]
        mylist = [0 if math.isnan(x) else x for x in sentence_vector]
        if (max(mylist) > threshold): #Threshold 
            emotionalities.append(sentence_vectors[i].index(max(mylist)))
        else: 
            emotionalities.append(8)
    return emotionalities

def compute_sentiments(sentence_vectors_sent):
    sentiments = []
    threshold = 1.0
    for i in range(0,len(sentence_vectors_sent)):
        sentence_vector = sentence_vectors_sent[i]
        mylist = [0 if math.isnan(x) else x for x in sentence_vector]
        if (max(mylist) > threshold): #Threshold 
            sentiments.append(sentence_vectors_sent[i].index(max(mylist)))
        else: 
            sentiments.append(2)
    return sentiments