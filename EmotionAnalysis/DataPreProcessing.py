import re
import io
from nltk.tokenize import RegexpTokenizer

import nltk.tag.stanford as st
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

# Remove Tweets where one of those columns is NAN:
# 'createdAt', 'text'
# 'longitude', 'latitude', 'placeLongitude', 'placeLatitude'
# TODO: Remove Tweets for which geolocation does not correspond to Switzerland
def clean_tweets(sample_tweets):
sample_tweets_cleaned = sample_tweets[sample_tweets['createdAt'].notnull() & sample_tweets['text'].notnull() & (((
                         sample_tweets[
                             'longitude'].notnull() &
                         sample_tweets[
                             'latitude'].notnull()) & ((
                         sample_tweets[
                             'longitude'] != '\N') & (
                         sample_tweets[
                             'latitude'] != '\N'))) | (
                        (
                        sample_tweets[
                            'placeLongitude'].notnull() &
                        sample_tweets[
                            'placeLatitude'].notnull()) & (
                        (
                        sample_tweets[
                            'placeLongitude'] != '\N') & (
                        sample_tweets[
                            'placeLatitude'] != '\N'))))]

return sample_tweets_cleaned


# Handling Entities/ Special categories:
#   Replacing @ instances with <username>
#   Replacing urls with <url>
#   TODO: Replacing Emoticons with their word meaning
#   Replacing numbers/phone/fax with <number>
#   TODO: Detecting place / city / country / any geolocation cues in any part of the tweet (Create a dictionary of places/cities in Switzerland)
def handle_special_categories(sample_tweets):
    tweets_list = []
    for i in range(0, len(sample_tweets)):
        #new_text = re.sub(r"http\S+", "<url>", sample_tweets.iloc[i]['text'])
        #new_text = re.sub(r"@\S+", "<username>", sample_tweets.iloc[i]['text'])
        #new_text = re.sub(r"\d+", "<number>", sample_tweets.iloc[i]['text'])
        dict = {
            r"http\S+": "<url>",
            r"@\S+": "<username>",
            r"\d+": "<number>",
            r"#": ""
        }
        tweets_list.append(re.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], sample_tweets.iloc[i]['text']))
    sample_tweets['text'] = tweets_list


# Tokenization and replacing contractions
def bag_of_word_representation(sample_tweets):
    """
    Tokenization, UTF-8 decoding and Removal of white spaces
    :param tweets:
    :return:
    """
    tweets_bag_words = []
    tokenizer = RegexpTokenizer(r'\w+')
    for tweet in sample_tweets['Subtitle']:
        # Tokenization
        tweets_filtered = [t for t in tokenizer.tokenize(tweet)]
        tweets_bag_words.append(tweets_filtered)
    return tweets_bag_words

def replace_contractions(sample_tweets):
    f = io.open('/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis/contractions.txt', 'r',
                encoding='utf8')
    text = f.read()
    contractions = eval(text)
    keys = list(contractions.keys())
    values = list(contractions.values())
    for i in range(0, len(contractions)):
        sample_tweets = sample_tweets.replace({keys[i]: values[i]}, regex=True)
    return sample_tweets

# Part of Speech Tagging to recognize Affective words (Noun, Verbs, Adjectives, Adverbs)
def pos_tagging(tweets_bag_words):
    """
    POS tagging of tweets using universal tagset
    :param tweets_bag_words:
    :return:
    """
    tagged_tweets = []
    for i in range(0, len(tweets_bag_words)):
        tagged_tweets.append(nltk.pos_tag(tweets_bag_words[i]))
    return tagged_tweets

def extract_Affect_words(tagged_tweets):
    """
    :param tagged_tweets:
    :return: tweets_nava
    """
    tweets_nava = []
    tweets_nava_sub = []
    for i in range(0, len(tagged_tweets)):
        for (word, tag) in tagged_tweets[i]:
            if tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS':
                tweets_nava_sub.append((word, 'N'))
            elif tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                tweets_nava_sub.append((word, 'V'))
            elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                tweets_nava_sub.append((word, 'Adj'))
            elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                tweets_nava_sub.append((word, 'Adv'))
        tweets_nava.append(list(tweets_nava_sub))
        tweets_nava_sub = []
    return tweets_nava


# Some NRE to replace basic entities like Proper Nouns with tag <proper_noun>
def nre_tagging(tweets_nava):
    """
    :param tweets_bag_words:
    :return: nre_tagged
    """
    tagger = st.StanfordNERTagger(
        '/home/meryem/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz',
        '/home/meryem/stanford-ner-2014-06-16/stanford-ner.jar')

    nre_tagged = []
    for i in range(0, len(tweets_nava)):
        nre_tagged.append(tagger.tag(tweets_nava[i]))
    return nre_tagged

# TODO: FURTHER PRE-PROCESSING
# TODO: (possibly spell-checking as well)
# TODO: Lowering multiple occurences of a character in a word (words like soooooo => so)
# TODO: Lemmatization and term normalization to get less variable versions of the same word. (possibly use thesaurus also)
# TODO: Remove less frequent words => word count + define a specific threshold