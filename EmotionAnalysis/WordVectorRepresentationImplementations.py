# This file portrayed different implementations of word 2 vector representation:
    # Word2Vec
    # FastText
import numpy as np  # Make sure that numpy is imported
from tqdm import tqdm

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # sentence
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            if word in model:
                featureVec = np.add(featureVec,model[word])
                
    # 
    # Divide the result by the number of words to get the average
    if nwords != 0:
        featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(sentences, model, num_features):
    # Given a set of sentences (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    sentenceFeatureVecs = np.zeros((len(sentences),num_features),dtype="float32")
    # 
    # Loop through the sentences
    for i in tqdm(range(0,len(sentences))):
        # 
        # Call the function (defined above) that makes average feature vectors
        sentenceFeatureVecs[counter] = makeFeatureVec(sentences[i], model, num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return sentenceFeatureVecs
