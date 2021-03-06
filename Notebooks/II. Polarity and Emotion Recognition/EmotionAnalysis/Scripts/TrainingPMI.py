import numpy as np
import pandas as pd
import sys
from EmotionAnalysis.DataPreProcessing import *
from EmotionAnalysis.SentSemanticModule import *
import ast

print "Preparing Dataset"
#tweets_df1 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation1.csv',encoding='utf-8')
#tweets_df2 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation2.csv',encoding='utf-8')
#tweets_df3 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation3.csv',encoding='utf-8')
#tweets_df4 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation4.csv',encoding='utf-8')
#tweets_df5 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation5.csv',encoding='utf-8')
#tweets_df6 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation6.csv',encoding='utf-8')
#tweets_df7 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation7.csv',encoding='utf-8')
#tweets_df8 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Representation8.csv',encoding='utf-8')

#frames = [tweets_df1,tweets_df2,tweets_df3,tweets_df4,tweets_df5,tweets_df6,tweets_df7,tweets_df8]
#whole_df = pd.concat(frames)
#whole_df.to_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Repr_Whole.csv',encoding='utf-8',index=False)

whole_df = pd.read_csv('../../Project_Backup/BigData/Unannotated_Representation_whole.csv')
tokenized_lemma = whole_df['Tokenized Lemmatized']

# Convert tokenized_lemma
tokenized_lemmatized_tweets = []
for i in range(0, len(tokenized_lemma)):
    result = ast.literal_eval(tokenized_lemma[i])
    tokenized_lemmatized_tweets.append(result)

flatten_list = [word for sublist in tokenized_lemmatized_tweets for word in sublist]
print flatten_list[0]

print "Extracting Lexicon"
lexicon_df = pd.read_csv('NRCLexicon/lexicon_nrc.csv', encoding='utf-8')
unique_lexicon = make_unique_lexicon(lexicon_df)


print "Training PMI dictionary"
pmi_dict = calculate_pmi(flatten_list, unique_lexicon)
np.save('Models/pmi_dict.npy', pmi_dict)

# Load
pmi_dict_read = np.load('Models/pmi_dict.npy').item()
#print pmi_dict_read
