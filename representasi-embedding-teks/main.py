# AUTHOR: Rizal Setya Perdana (rizalespe@ub.ac.id)
# This code written for showing the process of generating embedding 
# representation of text data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import csv
import pickle
from helper import Vocabulary, TextPreprocess

"""
Datasource example:
https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_tweet_sentiment_pilkada_DKI_2017.csv
"""

datasource = 'dataset_tweet_sentiment_pilkada_DKI_2017.csv'
minimum_treshold = 5


with open(datasource) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    tweet_collection = []

    # Args: list of document, contain_header(True/False)
    # Return: text vocabulary with index
    for row in csv_reader:
        text_tweet = row[3]
        tweet_collection.append(text_tweet)
    
    """Generating the vocabulary file (index, word) from a csv file
    """
    Vocabulary().generate(list_document= tweet_collection, 
                            threshold=minimum_treshold,
                            contain_header=True,
                            save_to_file='vocab.pkl')

    """Mapping list of document to index based on the vocabulary file
    """
    maps = Vocabulary().map(vocabulary_file='vocab.pkl', 
                            list_document=tweet_collection,
                            contain_header=True)
    
    vocabulary_file= 'vocab.pkl'
    with open(vocabulary_file, 'rb') as f:
        vocab = pickle.load(f) 
    vocab_size = len(vocab)
    embed = nn.Embedding(vocab_size, embedding_dim=5)
    
    for x in maps:
        y = embed(torch.LongTensor(x))
        print(y.shape)

# make embedding representation of text document
# apply to RNN

