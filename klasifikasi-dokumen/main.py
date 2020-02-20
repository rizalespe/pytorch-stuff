import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import csv
import pickle
from helper import Vocabulary, TextPreprocess
from models import AnalisisSentimen

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

file_text = 'dataset_tweet_sentiment_pilkada_DKI_2017.csv'
minimum_treshold = 5
learning_rate = 0.001
num_epochs = 10
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(file_text) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    tweet_collection = []

    # Args: list of document, contain_header(True/False)
    # Return: text vocabulary with index
    for row in csv_reader:
        text_tweet = row[3]
        sentiment_class = row[1]
        tweet_collection.append([text_tweet, sentiment_class])
    
    
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
    document_map = Vocabulary().map(vocabulary_file=vocabulary_file, 
                            list_document=tweet_collection,
                            contain_header=True)
    
    
    with open(vocabulary_file, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print("Jumlah kata yang ada pada vocabulary: ", vocab_size)

   
    
    
    list_docs = []
    list_class = []
    for x in document_map:
        list_docs.append(torch.LongTensor(x[0]))
        if x[1] == 'positive':
            list_class.append(torch.LongTensor([1]))
        else:
            list_class.append(torch.LongTensor([0]))

    list_docs = rnn_utils.pad_sequence(list_docs, batch_first=True)
    data_loader = zip(list_docs, list_class)

     # model initialization
    model = AnalisisSentimen(vocab_size=vocab_size, embed_size=512, num_class=2, document_length= list_docs.shape[1]).to(device)
   
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(list_docs)
    for epoch in range(num_epochs):
        print("-------------------------------------------------") 
        for i, (document, labels) in enumerate(data_loader):
            
            model.train()
            document = document.to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = model(document)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(i)
                print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f} '.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
