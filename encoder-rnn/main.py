import torch
from helper import Vocabulary, TextPreprocess
import csv
import nltk
from collections import Counter
import pickle



"""
Datasource:
https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_tweet_sentiment_pilkada_DKI_2017.csv
"""
# read text file
datasource = 'dataset_tweet_sentiment_pilkada_DKI_2017.csv'
minimum_treshold = 10

with open(datasource) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    counter = Counter()
    for row in csv_reader:
        id_code     = row[0]
        sentiment   = row[1]
        calon       = row[2]
        text_tweet  = row[3]
        if line_count !=0: #skip the line with CSV column names (header)
            text_tweet = TextPreprocess().unicodeToAscii(text_tweet)
            text_tweet = TextPreprocess().normalizeString(text_tweet)
            try:
                tokens = nltk.tokenize.word_tokenize(text_tweet.lower())
            except Exception:
                pass
            counter.update(tokens)
        line_count+=1
    
    # Filtering low number occurence
    words = []
    for word, word_count in counter.items():
        if word_count >= minimum_treshold:
            words.append(word)

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    
        
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to 'vocab.pkl'")

    
      
# make embedding representation of text document
# apply to RNN

