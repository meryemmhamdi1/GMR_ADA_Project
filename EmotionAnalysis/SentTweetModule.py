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
    for e in range(0, 8):
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
            # Neutral emotion = index 8
            dominant_emotion_tweets.append(8)
        return emotion_dict.columns.values[dominant_emotion_tweets]

