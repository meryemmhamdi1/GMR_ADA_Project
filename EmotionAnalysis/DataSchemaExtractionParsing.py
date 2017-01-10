import pandas as pd


def extract_schema():
    schema = pd.read_csv('Data/schema.txt', engine='python', names=['col'])
    schema['ColumnID'] = schema['col'].str.split().apply(lambda x: x[0])
    schema['ColumnName'] = schema['col'].str.split().apply(lambda x: x[1])
    schema['DataType'] = schema['col'].str.split().apply(lambda x: x[2])
    schema['Signed/Unsigned'] = schema['col'].str.split().apply(lambda x: x[3])
    schema['Criteria1'] = schema['col'].str.split().apply(lambda x: x[4])
    schema['Criteria2'] = schema['col'].str.split().apply(lambda x: x[-1])
    del (schema['col'])
    return schema


def extract_tweets(location):
    schema = extract_schema()
    tweets = pd.read_excel(location, header=None)
    tweets.columns = list(schema['ColumnName'])
    return tweets

def extractLexicon():
    lexicon = pd.read_csv(
        '../NRCLexicon/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon.txt',
        engine='python', names=['col'])
    lexicon['Word'] = lexicon['col'].str.split().apply(lambda x: x[0])
    lexicon['EmotionCategory'] = lexicon['col'].str.split().apply(lambda x: x[1])
    lexicon['Score'] = lexicon['col'].str.split().apply(lambda x: x[-1])
    del (lexicon['col'])
    return lexicon
