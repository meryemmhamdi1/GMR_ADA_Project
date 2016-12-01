def average_words(tweets_nava, word_emotional_vectors_dict):
    matrix_sentences_list = []
    for i in range(0,len(tweets_nava)): # iterate over all sentences
        w, h = len(tweets_nava[i]), 8
        matrix_sentence = [[0 for x in range(w)] for y in range(h)]
        k = 0
        for (word, tag) in tweets_nava[i]:
            j = 0
            for e in range(0, 8): # Iterate over all emotions => fill in the emotional vectors for all words
                word_emotional_vectors_dict[word][e]
                matrix_sentence[j][k] = np.power(total_pmi, 1 / r)
                k += 1  # increment index of tweets_words
            matrix_sentences_list.append(matrix_sentence)
