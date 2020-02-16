# AUTHOR: Rizal Setya Perdana (rizalespe@ub.ac.id)
# This code is custom helper for text processing with the following purpose:
# 1. Vocabulary Generator class
# 2. Text Preprocessing class 

import unicodedata
import re
import string
from collections import Counter
import pickle
import nltk
import os

class Vocabulary(object):
    """This class is a custom data structure object with the basic functional
    such as adding new word, retrieving the index of word, and find the length
    of the 
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def generate(self, list_document, threshold=10, contain_header=False, 
                save_to_file= 'vocab.pkl'):
        
        # Removing the first element of list if this is a header, default is 
        # False
        if contain_header == True:
            list_document.pop(0)
        counter = Counter()
        
        for document in list_document:
            document = TextPreprocess().unicodeToAscii(document)
            document = TextPreprocess().normalizeString(document)
            try:
                tokens = nltk.tokenize.word_tokenize(document.lower())
            except Exception:
                pass
            counter.update(tokens)
        
        # Filtering out low number occurence which less then the threshold
        words = []
        for word, word_count in counter.items():
            if word_count >= threshold:
                words.append(word)
        
        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        for i, word in enumerate(words):
            vocab.add_word(word)
        

        # saving the word with the index in pickle file format
        with open(save_to_file, 'wb') as f:
            pickle.dump(vocab, f)
    
    def map(self, vocabulary_file, list_document, contain_header=False):
        # Translate from text document to the specified vocabulary file
        if not os.path.exists(vocabulary_file):
            raise RuntimeError('Vocabulary file ', vocabulary_file, 'not found') 
        
        with open(vocabulary_file, 'rb') as f:
            vocab = pickle.load(f) 
        # Removing the first element of list if this is a header, default is 
        # False
        if contain_header == True:
            list_document.pop(0)

        list_document_map = []
        documents = []
        for document in list_document:
            document = TextPreprocess().unicodeToAscii(document)
            document = TextPreprocess().normalizeString(document)
            document = '<start> ' +document+ ' <end>'
            document = document.split()
            documents.append(document)
            tokens = []
            for token in document:
                token = vocab(token)
                tokens.append(token)            
            list_document_map.append(tokens)
    
        return list_document_map

class TextPreprocess:
    def __init__(self):
        pass
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        # Remove the punctuation
        s = s.translate(str.maketrans('', '', string.punctuation))
        # remove white space before and after document
        s = s.strip() 
        return s