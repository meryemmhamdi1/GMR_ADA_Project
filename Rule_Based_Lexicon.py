import pandas as pd
import sys
#sys.path.insert(0, "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis")
from EmotionAnalysis.DataSchemaExtractionParsing import *
from EmotionAnalysis.DataPreProcessing import *
from EmotionAnalysis.SentSemanticModule import *
from EmotionAnalysis.SentTweetModule import *
import ast
from tqdm import tqdm
import numpy as np
import pickle



#if __name__ == '__main__':

###### STEP 1: Loading Data with tokenized and affective representation:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA

## TO BE REPLACED:

tweets_df = pd.read_csv('../../Project_Backup/BigData/Unannotated_Representation/pt/Unannotated_Representation1_sw.csv',nrows=100)

#tweets_df = tweets_df[1000000:len(tweets_df)]

nava_repr = list(tweets_df['Nava without Stop Words'])

print ("Converting NAVA Tweets to List")
# Convert nava_tweets
nava_tweets = []
for i in range(0, len(nava_repr)):
    result = ast.literal_eval(nava_repr[i])
    nava_tweets.append(result)
    #nava_tweets.append(nava_repr[i][1:len(nava_repr[i])-1].split(', '))

print ("Reading NRC Lexicon")
###### STEP 2: Loading Lexicon:
## TO BE REPLACED:
#lexicon_df = pd.read_csv('NRCLexicon/lexicon_nrc.csv', encoding='utf-8')
#lexicon_dict = np.load('NRCLexicon/french_lexicon.npy').item()

with open('NRCLexicon/turkish_lexicon.pickle', 'rb') as handle:
    lexicon_dict = pickle.load(handle)

###### STEP 3: Word Level
matrix_sentences_lexicon = compute_matrix_sentences_list_lexicon(nava_tweets,lexicon_dict)

print ("Finished Computing Matrix Sentences")
###### STEP 4: Sentence Level:

# Emotion Recognition
print ("Emotion Recognition")
sentence_vectors_lexicon = compute_sentence_emotion_vectors(matrix_sentences_lexicon)

print ("Computing Emotionalities")
emotionalities = compute_emotionalities(sentence_vectors_lexicon)


# Sentiment Analysis
print ("Sentiment Analysis")
sentence_vectors_sent = compute_sentence_sentiment_vectors(matrix_sentences_lexicon)

print ("Computing Sentiments")
sentiments = compute_sentiments(sentence_vectors_sent,emotionalities)

###### FINAL STEP 5: Storing Emotion + Sentiment for each tweet

emo_dict = {
    0: 'Anger',
    1: 'Anticipation',
    2: 'Disgust',
    3: 'Fear',
    4: 'Joy',
    5: 'Sadness',
    6: 'Surprise',
    7: 'Trust',
    8: 'Neutral'
}
sent_dict = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


print ("Storing in dataframe")

emotions = []
senti = []
for i in range(0,len(emotionalities)):
    emotions.append(emo_dict[emotionalities[i]])
    senti.append(sent_dict[sentiments[i]])

lexicon_results_df = pd.DataFrame()

lexicon_results_df['Nava Tweet'] = nava_tweets

lexicon_results_df['Emotion'] = emotions

lexicon_results_df['Emotion Vectors'] = sentence_vectors_lexicon

lexicon_results_df['Sentiment'] = senti

lexicon_results_df['Sentiment Vectors'] = sentence_vectors_sent

lexicon_results_df.to_csv('../../Project_Backup/BigData/LexiconBasedResults/pt/Tweets_Labelled_Lexicon_sample.csv')

