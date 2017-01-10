import re
import io
from nltk.tokenize import RegexpTokenizer

import nltk.tag.stanford as st
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

# Remove Tweets where one of those columns is NAN:
# 'createdAt', 'text'
# 'longitude', 'latitude', 'placeLongitude', 'placeLatitude'
# TODO: Remove Tweets for which geolocation does not correspond to Switzerland
def clean_tweets(sample_tweets):
    sample_tweets_cleaned = sample_tweets[sample_tweets['createdAt'].notnull() &sample_tweets['text'].notnull() & (((sample_tweets['longitude'].notnull() &sample_tweets['latitude'].notnull())) |((sample_tweets['placeLongitude'].notnull()&sample_tweets['placeLatitude'].notnull())))]
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
        new_text = re.sub(r"http\S+", "", sample_tweets.iloc[i]['text'])
        new_text = re.sub(r"@\S+", "", new_text)
        new_text = re.sub(r"\d+", "", new_text)
        new_text = re.sub(r"#", "", new_text)
        tweets_list.append(new_text)
    sample_tweets['text'] = tweets_list
    return sample_tweets

def replace_contractions(sample_tweets):
    f = io.open('contractions.txt', 'r',
                encoding='utf8')
    text = f.read()
    contractions = eval(text)
    keys = list(contractions.keys())
    values = list(contractions.values())
    for i in range(0, len(contractions)):
        sample_tweets = sample_tweets.replace({keys[i]: values[i]}, regex=True)
    return sample_tweets

# Detecting Ascii characters
def is_ascii(s):
    return all (ord(c) < 128 for c in s)

# Tokenization and replacing contractions
def bag_of_word_representation(sample_tweets):
    """
    Tokenization, UTF-8 decoding and Removal of white spaces
    :param tweets:
    :return:
    """
    tweets_bag_words = []
    tokenizer = RegexpTokenizer(r'\w+')
    for tweet in sample_tweets['text']:
        # Removing of non-ascii character
        non_ascii_tweet = re.sub(r'[^\x00-\x7F]+','',tweet)
        # tweets_filtered = [word for word in tweet if is_ascii(str(word))]  
        # Tokenization
        tweets_tokenized = [t for t in tokenizer.tokenize(non_ascii_tweet)]
        tweets_bag_words.append(tweets_tokenized)
    return tweets_bag_words
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

def normalize_pos_tags_words(tagged_tweets):
    """
    :param tagged_tweets:
    :return: tweets_nava
    """
    tweets_nava = []
    tweets_nava_sub = []
    for i in range(0, len(tagged_tweets)):
        for (word, tag) in tagged_tweets[i]:
            if tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS':
                tweets_nava_sub.append((word, 'n'))
            elif tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                tweets_nava_sub.append((word, 'v'))
            elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                tweets_nava_sub.append((word, 'Adj'))
            elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                tweets_nava_sub.append((word, 'Adv'))
            else: 
                tweets_nava_sub.append((word, tag))
        tweets_nava.append(list(tweets_nava_sub))
        tweets_nava_sub = []
    return tweets_nava

def normalize_pos_tags_words1(tagged_tweets):
    """
    :param tagged_tweets:
    :return: tweets_nava
    """
    tweets_nava = []
    tweets_nava_sub = []
    for i in range(0, len(tagged_tweets)):
        for (word, tag) in tagged_tweets[i]:
            if tag == "NOUN":
                tweets_nava_sub.append((word, 'n'))
            elif tag == 'VERB':
                tweets_nava_sub.append((word, 'v'))
            else: 
                tweets_nava_sub.append((word, tag))
        tweets_nava.append(list(tweets_nava_sub))
        tweets_nava_sub = []
    return tweets_nava

def keep_only_nava_words(tagged_tweets):
    """
    :param tagged_tweets:
    :return: tweets_nava
    """
    tweets_nava = []
    tweets_nava_sub = []
    for i in range(0, len(tagged_tweets)):
        for (word, tag) in tagged_tweets[i]:
            if tag == "n" or tag == "v" or tag =="ADJ" or tag == "ADV":
                tweets_nava_sub.append(word)
        tweets_nava.append(list(tweets_nava_sub))
        tweets_nava_sub = []
    return tweets_nava

# Some NRE to replace basic entities like Proper Nouns with tag <proper_noun>
def extract_entity_names(t):
    non_entity_names = []
    entity_names = []
   
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    else:
        non_entity_names.append(t)
    return non_entity_names

def remove_named_entities(new_samples):
    tweet_without_ne = []
    for tweet in new_samples:
        nre_tweet = nltk.ne_chunk(tweet, binary = True)
        non_entity_names = []
        for tree in nre_tweet:    
            non_entity_names.extend(extract_entity_names(tree))
        tweet_without_ne.append(non_entity_names)
    return tweet_without_ne

# TODO: FURTHER PRE-PROCESSING
# TODO: (possibly spell-checking as well)
# TODO: Lowering multiple occurences of a character in a word (words like soooooo => so)
# TODO: Lemmatization and term normalization to get less variable versions of the same word. (possibly use thesaurus also)
def lemmatizer(tweets):
    tweets_whole = []
    lmtzr = WordNetLemmatizer()
    for i in range(0,len(tweets)):
        tweets_sub = []
        for (word,tag) in tweets[i]:
            if tag=='v' or tag =='n':
                tweets_sub.append((lmtzr.lemmatize(word,tag),tag))
            else: 
                tweets_sub.append((word,tag))
        tweets_whole.append(tweets_sub)
    return tweets_whole


def lemmatizer_untagged(tweets):
    tweets_whole = []
    lmtzr = WordNetLemmatizer()
    for i in range(0,len(tweets)):
        tweets_sub = []
        for (word,tag) in tweets[i]:
            if tag=='v' or tag =='n':
                tweets_sub.append(lmtzr.lemmatize(word,tag))
            else: 
                tweets_sub.append(word)
        tweets_whole.append(tweets_sub)
    return tweets_whole

def lemmatizer_raw(tweets):
    tweets_whole = []
    lmtzr = WordNetLemmatizer()
    for i in range(0,len(tweets)):
        tweets_sub = []
        for (word,tag) in tweets[i]:
            if tag=='v' or tag =='n':
                tweets_sub.append(unicode(lmtzr.lemmatize(word,tag)).lower())
            else: 
                tweets_sub.append(unicode(word.lower()))
        tweets_whole.append(tweets_sub)
    return tweets_whole

# TODO: Remove less frequent words => word count + define a specific threshold

def eliminate_stop_words_punct(tagged_tweets):
    """
    Elimination of Stop words
    Elimination of Punctuation

    :rtype: object
    :param tagged_tweets:
    :return: tagged_tweets_without
    """
    stop_words = list(set(stopwords.words('english')))
    non_emotinal_verbs = ['go','be','do','have','get']
    customized_stop_words = stop_words + non_emotinal_verbs
    tagged_tweets_without = []
    for i in range(0, len(tagged_tweets)):
        tagged_tweets_without_sub = []
        for (word, tag) in tagged_tweets[i]:
            if word not in customized_stop_words and word not in ['url','number','username'] and len(word) >= 2:
                tagged_tweets_without_sub.append((word.lower(), tag))
        tagged_tweets_without.append(tagged_tweets_without_sub)
    return tagged_tweets_without

def make_unique(duplicate_list):
    """

    :param duplicate_list:
    :return:
    """
    unique_words = list(set(duplicate_list))
    return unique_words

def make_unique_lexicon(nrc_lexicon):
    """

    :param nrc_lexicon:
    :return: unique_lexicon
    """
    lexicon_flatten = []
    emotions = nrc_lexicon.columns.values
    for i in range(0, len(emotions)):
        for representative_word in nrc_lexicon[emotions[i]].dropna():
            lexicon_flatten.append(representative_word)
    unique_lexicon = make_unique(lexicon_flatten)
    return unique_lexicon

def list_nrc_lexicon(nrc_lexicon):
    """

    :param nrc_lexicon:
    :return: sm_list
    """
    emotions = nrc_lexicon.columns.values
    sm_list = []
    for emotion in emotions:
        sm = list(nrc_lexicon[emotion].dropna())
        sm_list.append(sm)
    return sm_list