import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/EmotionAnalysis")
from DataPreProcessing import *
from SEMProjectSemanticModule import *
import ast


whole_df = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Unannotated_Repr_Whole.csv')
tokenized_lemma = whole_df['Tokenized Lemmatized']

# Convert tokenized_lemma
tokenized_lemmatized_tweets = []
for i in range(0, len(tokenized_lemma)):
    result = ast.literal_eval(tokenized_lemma[i])
    tokenized_lemmatized_tweets.append(result)

print tokenized_lemmatized_tweets[0]
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 1  # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
sentences = [ "the quick brown fox jumps over the lazy dogs",
"Then a cop quizzed Mick Jagger's ex-wives briefly." ]
model = word2vec.Word2Vec([s.encode('utf-8').split() for s in sentences], workers=num_workers,
            size=num_features, min_count = min_word_count,
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/Models/300features_40minwords_10context"
model.save(model_name)

# Trying model
# model.accuracy('/home/meryem/Dropbox/meryem/dataset/BBT_Transcripts/BBT - 01x01 - Pilot Episode')
model.similarity('quick','over')