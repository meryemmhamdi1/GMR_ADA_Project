import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from EmotionAnalysis.DataPreProcessing import *

from tqdm import tqdm
import ast
from pattern.fr import parse, lemma

tweets_df = pd.read_csv('../../Project_Backup/BigData/RawData/fr_5.csv',encoding = "ISO-8859-1")

print ("SOME CLEANING >>>>>")
replaced_categories = handle_special_categories(tweets_df)

print ("BAG OF WORD REPRESENTATION >>>>>")
tokenized_list = bag_of_word_representation(replaced_categories)

text_tweets = []
for i in range(0,len(tokenized_list)):
    space = ' '
    text_tweets.append(space.join(tokenized_list[i]))

print ("Parsing and Lemmatizing")
tokenized_tweets = []
tokenized_tweets_untag = []
for i in tqdm(range(0,len(text_tweets))):
    #print tokenized_tweets[i]
    tree = parse(text_tweets[i],lemmata=True).split()
    tokenized_tweets_sub = []
    tokenized_tweets_sub_untag = []
    if len(text_tweets[i])>1:
        for i in range(0,len(tree[0])):
            tokenized_tweets_sub.append((tree[0][i][4],tree[0][i][1]))
            tokenized_tweets_sub_untag.append(tree[0][i][4])
    tokenized_tweets.append(tokenized_tweets_sub)
    tokenized_tweets_untag.append(tokenized_tweets_sub_untag)

print ("Keeping only Nava words")
nava_tweets= keep_only_nava_words(normalize_pos_tags_words(tokenized_tweets))
print ("Removing Stopwords")
stop_words = list(set(stopwords.words('french')))
nava_tweets_sw = []
for i in tqdm(range(0, len(nava_tweets))):
    nava_tweets_sw.append([t for t in nava_tweets[i] if t not in stop_words])

print ("Storing in DataFrame")
tweets = pd.DataFrame()
tweets['Tokenized Lemmatization'] = tokenized_tweets_untag
tweets['Nava without Stop Words'] = nava_tweets_sw
tweets.to_csv('../../Project_Backup/BigData/Unannotated_Representation/fr/Unannotated_Representation1_5.csv',encoding = "ISO-8859-1")

