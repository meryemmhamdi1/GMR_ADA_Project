import pandas as pd
import sys
#sys.path.insert(0, "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis") 
from DataSchemaExtractionParsing import *
from DataPreProcessing import *
from SentSemanticModule import *
from SentTweetModule import *
from SentSyntacticModule import *
import ast

###### STEP 1: Loading Data with tokenized and affective representation:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA

## TO BE REPLACED:

tweets_df = pd.read_csv('../Results/Test/Unannotated_Representation.csv')

nava_repr = tweets_df['Nava Representation']

# Convert nava_tweets 
nava_tweets = []
for i in range(0, len(nava_repr)):
    result = ast.literal_eval(nava_repr[i])
    nava_tweets.append(result)


###### STEP 2: Loading Lexicon:
## TO BE REPLACED:
lexicon_df = pd.read_csv('../NRCLexicon/lexicon_nrc.csv', encoding='utf-8')
unique_lexicon = make_unique_lexicon(lexicon_df)

###### STEP 3: Word Level
matrix_sentences_lexicon = compute_matrix_sentences_list_lexicon(nava_tweets,lexicon_df)

print "Finished Computing Matrix Sentences"
###### STEP 4: Sentence Level:
 
# Emotion Recognition
sentence_vectors_lexicon = compute_sentence_emotion_vectors(matrix_sentences_lexicon)

emotionalities = compute_emotionalities(sentence_vectors_lexicon)


# Sentiment Analysis
sentence_vectors_sent = compute_sentence_sentiment_vectors(matrix_sentences_lexicon)

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

emotions = []
senti = []
for i in range(0,len(emotionalities)):
    emotions.append(emo_dict[emotionalities[i]])
    senti.append(sent_dict[sentiments[i]])

lexicon_results_df = pd.DataFrame()

lexicon_results_df['Nava Tweet'] = nava_tweets

lexicon_results_df['Emotion'] = emotions

lexicon_results_df['Sentiment'] = senti

lexicon_results_df.to_csv('../Results/Test/Tweets_Labelled_Lexicon.csv')

