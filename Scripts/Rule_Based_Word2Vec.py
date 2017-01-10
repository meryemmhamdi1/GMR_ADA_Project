import pandas as pd
import sys
sys.path.insert(0, "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis") 
from DataSchemaExtractionParsing import *
from DataPreProcessing import *
from SentSemanticModule import *
from SentTweetModule import *
from SentSyntacticModule import *
import ast

##### STEP 1: Loading Data with tokenized and affective representation:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA

tweets_df = pd.read_csv('/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Results/Test/Unannotated_Representation.csv')

tokenized_lemma = tweets_df['Tokenized Lemmatized']

# Convert tokenized_lemma
tokenized_lemmatized_tweets = []
for i in range(0, len(tokenized_lemma)):
    result = ast.literal_eval(tokenized_lemma[i])
    tokenized_lemmatized_tweets.append(result)

nava_repr = tweets_df['Nava Representation'] 

# Convert nava_tweets 
nava_tweets = []
for i in range(0, len(nava_repr)):
    result = ast.literal_eval(nava_repr[i])
    nava_tweets.append(result)

###### STEP 2: Loading Lexicon:
lexicon_df = pd.read_csv('/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/NRCLexicon/lexicon_nrc.csv',encoding='utf-8')
unique_lexicon = make_unique_lexicon(lexicon_df)

###### STEP 3: Loading Word2Vec Model:

model = Word2Vec.load('/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Models/geo_tweets_word2vec_model')

###### STEP 4: Word Level

matrix_sentences_word2vec = compute_matrix_sentences_list_word2vec(lemmatized_tweets,lexicon_df,model)


###### STEP 5: Sentence Level:
 
# Emotion Recognition
sentence_vectors_word2vec = compute_sentence_emotion_vectors(matrix_sentences_word2vec)

emotionalities = compute_emotionalities(sentence_vectors_word2vec)


# Sentiment Analysis
sentence_vectors_sent = compute_sentence_sentiment_vectors(sentence_vectors_word2vec)

sentiments = compute_sentiments(sentence_vectors_sent,emotionalities)

###### FINAL STEP 6: Storing Emotion + Sentiment for each tweet

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
    0: "Positive",
    1: "Negative",
    2: "Neutral"
}

emotions = []
senti = []
for i in range(0,len(emotionalities)):
    emotions.append(emo_dict[emotionalities[i]])
    senti.append(sent_dict[sentiments[i]])

word2vec_results_df = pd.DataFrame()

word2vec_results_df['Nava Tweet'] = nava_tweets

word2vec_results_df['Emotion'] = emotions

word2vec_results_df['Sentiment'] = senti

word2vec_results_df.to_csv('/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Results/Test/Tweets_Labelled_Word2Vec.csv')

