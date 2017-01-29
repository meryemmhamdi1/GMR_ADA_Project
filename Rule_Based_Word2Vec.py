import pandas as pd
import sys
sys.path.insert(0, "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis")
from EmotionAnalysis.SentSemanticModule import *
from EmotionAnalysis.SentTweetModule import *
import ast
from gensim.models import Word2Vec


##### STEP 1: Loading Data with tokenized and affective representation:

tweets_df = pd.read_csv('../../Project_Backup/BigData/Unannotated_Representation/en/Unannotated_Representation_part1.csv',nrows=10)

nava_repr = tweets_df['Nava Representation']

print ("Convert nava tweets")
# Convert nava_tweets 
nava_tweets = []
for i in tqdm(range(0, len(nava_repr))):
    nava_tweets.append(nava_repr[i][1:len(nava_repr[i])-1].split(', '))

print ("Loading Lexicon")
###### STEP 2: Loading Lexicon:
lexicon_df = pd.read_csv('NRCLexicon/lexicon_nrc.csv',encoding='utf-8')
unique_lexicon = make_unique_lexicon(lexicon_df)

###### STEP 3: Loading Word2Vec Model:
print ("Loading Word2Vec")
model = Word2Vec.load('../../Project_Backup/BigData/Models/whole_en_model')

###### STEP 4: Word Level
print ("Computing word level scores")
matrix_sentences_word2vec = compute_matrix_sentences_list_word2vec(nava_tweets,lexicon_df,model)


###### STEP 5: Sentence Level:
print ("Computing Emotionalities")
# Emotion Recognition
sentence_vectors_word2vec = compute_sentence_emotion_vectors(matrix_sentences_word2vec)

emotionalities = compute_emotionalities(sentence_vectors_word2vec)


# Sentiment Analysis
print ("Computing sentiments")
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

print ("Storing in dataframe")
word2vec_results_df = pd.DataFrame()

word2vec_results_df['Nava Tweet'] = nava_tweets

word2vec_results_df['Emotion'] = emotions

word2vec_results_df['Sentiment'] = senti

word2vec_results_df.to_csv('../Results/Test/Tweets_Labelled_Word2Vec1.csv')

