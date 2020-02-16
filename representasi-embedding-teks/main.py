# AUTHOR: Rizal Setya Perdana (rizalespe@ub.ac.id)
# This code written for showing the process of generating embedding 
# representation of text data

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
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
    
    print("Jumlah dokumen tweet dalam list: ", len(tweet_collection))
    
    """ Generating the vocabulary file (index, word) from a csv file and save 
        only word >= minimum threshold value
    """
    Vocabulary().generate(list_document= tweet_collection, 
                            threshold=minimum_treshold,
                            contain_header=True,
                            save_to_file='vocab.pkl')

    """Mapping list of document to index based on the vocabulary file
    """
    vocabulary_file= 'vocab.pkl'
    maps = Vocabulary().map(vocabulary_file=vocabulary_file, 
                            list_document=tweet_collection,
                            contain_header=True)
    
    
    with open(vocabulary_file, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print("Jumlah kata yang ada pada vocabulary: ", vocab_size)

    #instantiate embedding layer
    embed = nn.Embedding(vocab_size, embedding_dim=10)
    print("Ukuran layer embedding: ", embed)
    
    # generate list of document
    list_docs = []
    for x in maps:
        list_docs.append(torch.LongTensor(x))
    
    """Pad the sequences: proses ini meratakan dokumen yang memiliki panjang 
    kata berbeda-beda. Setelah melalui proses pad sequence ini, seluruh dokumen 
    pada corpus akan memiliki panjang yang sama.
    """
    list_docs = rnn_utils.pad_sequence(list_docs, batch_first=True)
    embedded_doc = embed(list_docs)
    print("Output embedding: ", embedded_doc.shape)
