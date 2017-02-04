import pandas as pd

from EmotionAnalysis.DataPreProcessing import *

#from EmotionAnalysis.SentSemanticModule import *
from pattern.fr import parse, lemma


###### STEP 1: Loading Data:
# HERE YOU CAN CHANGE THE NAME OF THE FILE FROM WHICH TO LOAD THE DATA
tweets = pd.read_csv("../../Project_Backup/BigData/RawData/fr.csv", encoding ="utf-8")
#print (len(tweets))
#tweets = tweets[1000000:len(tweets)]#:len(tweets)]
###### STEP 2: Replacing Special Categories:
print ("SOME CLEANING >>>>>")
replaced_categories = handle_special_categories(tweets)

###### STEP 3: Replacing contractions (needed for more accurate tokenization)
#tweets_no_contractions = replace_contractions(replaced_categories)

###### STEP 4: Tokenization of Tweets into words
print ("BAG OF WORD REPRESENTATION >>>>>")
tokenized_list = bag_of_word_representation(replaced_categories)

###### STEP 5: Part of Speech Tagging:
#print ("Part of Speech Tagging >>>>>")
#tagged_tweets = pos_tagging(tokenized_list)

###### STEP 6: Tokenized Lemmatized Representation:
print ("Lemmatization >>>>>")
#new_tagged = normalize_pos_tags_words(tagged_tweets)
#tokenized_lemma = lemmatizer_raw(new_tagged)
nava_tweets = []
for i in tqdm(range(0,len(tokenized_list))):
    #print tokenized_tweets[i]
    tree = parse(tokenized_list[i],lemmata=True).split()
    nava_tweets_sub = []
    if len(tokenized_list[i])>1:
        for i in range(0,len(tree[0])):
            nava_tweets_sub.append((tree[0][i][4],tree[0][i][1]))
    nava_tweets.append(nava_tweets_sub)

#print ("LOADING spaCy NLP >>>>>")
###### STEP 7: Loading spaCy:
#nlp = spacy.load('de')

#print ("Dependency Parsing >>>>>")
###### STEP 8: Dependency Parsing:
#docs = []
# Joining text:
#tweets_text = []
#for i in range(0, len(tokenized_list)):
#    space = u" "
#    tweets_text.append(space.join(tokenized_list[i]))
#tweets_text[0].encode("utf-8")
#for i in range(0, len(tweets_text)):
#    doc = nlp(tweets_text[i])
#    docs.append(doc)

#new_samples = []
#for sample in docs:
#    new_samples_sub = []
#    for word in sample:
#        new_samples_sub.append((unicode(word.lemma_),word.pos_))
#    new_samples.append(new_samples_sub)

#print ("Applying Syntactic Rules >>>>>>")

###### STEP 9: Applying Syntactic Rules:
#new_samples,triple_dependencies = apply_syntactic_rules_german(docs,new_samples)

print ("Further Cleaning >>>>>>>>>>")
###### STEP 10: Applying Named Entity Tagging:
#tweet_without_ne = remove_named_entities(tagged_tweets)


###### STEP 11: Normalizing POS tag:
normalized_tags = normalize_pos_tags_words(tagged_tweets)

###### STEP 12: Removal of Punctuation and Stop words and Converting to Lower Case and Removal of Other special categories: url, number, username:
tagged_tweets_without = eliminate_stop_words_punct(normalized_tags)

###### STEP 13: Lemmatization:
#lemmatized_tweets = lemmatizer(tagged_tweets_without)

###### STEP 14: Keeping only NAVA words:
nava_tweets = keep_only_nava_words(tagged_tweets_without)
print ("Storing in DataFrame")
###### STEP 16: Storing Tokenized Lemmatized + Affective Representation + Emotion for each tweet
tweets_df = pd.DataFrame()

print ("Storing in DataFrame")

tweets_df['Tokenized Lemmatized'] = tokenized_lemma
print ("Storing in DataFrame")

tweets_df['Nava Representation'] = nava_tweets
print ("Storing in DataFrame")


#print tokenized_lemma
tweets_df.to_csv('../../Project_Backup/BigData/Unannotated_Representation/de/Unannotated_Representation_sample.csv')

