import nltk
import pandas as pd
import numpy as np
import string
import torch
import torchtext
from torchdata.datapipes.iter import IterableWrapper

stemmer = nltk.stem.snowball.SnowballStemmer('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getDict(dataPipe):

    data_dict = {
        'Question': [],
        'Answer': []
    }
    
    for _, question, answers, _ in dataPipe:
        data_dict['Question'].append(question)
        data_dict['Answer'].append(answers[0])
        
    return data_dict


def loadDF(path):
    '''
    You will use this function to load the dataset into a Pandas Dataframe for processing.
    '''
    # load data
    train_data, val_data = torchtext.datasets.SQuAD1(path)
    
    # convert dataPipe to dictionary 
    train_dict, val_dict = getDict(train_data), getDict(val_data)
    
    # convert Dictionaries to Pandas DataFrame
    train_df = pd.DataFrame(train_dict)    
    validation_df = pd.DataFrame(val_dict)    
    
    return train_df.append(validation_df)


def tokenizer(sentence):
    '''
    Our text needs to be cleaned with a tokenizer. This function will perform that task.
    https://www.nltk.org/api/nltk.tokenize.html
    '''
    # clean text and tokenize it 
    #removes all punctuation marks from the text. 
    sentence = ''.join([s.lower() for s in sentence if s not in string.punctuation])
    
    # split sentences into words, then stem each word before joining back the words to sentence
    # Stemming is a process of reducing words to their base or root form.
    sentence = ' '.join(stemmer.stem(w) for w in sentence.split())
    sentence = ' '.join(w for w in sentence.split())
    
    # tokenize the sentence into words
    # consider using word_tokenize instead of regex to get hyphened words as well
    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(sentence)
    
    return sentence, tokens

# def toTensor(vocab, sentence):
#     # convert list of words "sentence" to a torch tensor of indices
#     indices = [vocab[word] for word in sentence.split(' ')]
#     # indices.append(vocab.word2index['<unk>', '<pad>'])
#     return add_symbols(vocab, torch.Tensor(indices).long()).view(-1, 1)
    

def getPairs(df):
    # convert df to list of pairs
    Q = df['Qtoken'].apply(lambda x: " ".join(x) ).to_list()
    A = df['Atoken'].apply(lambda x: " ".join(x) ).to_list()
    return [list(i) for i in zip(Q, A)]


def add_symbols(sentence, vocab):
    sos = torch.tensor([vocab['<sos>']])
    eos = torch.tensor([vocab['<eos>']])
    return torch.cat((sos,sentence,eos))

def add_symbols2(sentence, vocab):
    eos = torch.tensor([vocab['<eos>']])
    return torch.cat((sentence,eos))


def create_word_embedding(emb_dict, word_vocab):
    '''
    Creates a weight matrix of the words that are common in the brown vocab and
    the dataset's vocab. Initializes OOV words with a zero vector.
    '''
    weights_matrix = np.zeros((len(word_vocab), 256))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        except:
            pass
    return weights_matrix, words_found


# import cupy as cp
# import numpy as np

# def create_word_embedding(emb_dict, word_vocab):
#     '''
#     Creates a weight matrix of the words that are common in the brown vocab and
#     the dataset's vocab. Initializes OOV words with a zero vector.
#     '''
#     # Create a Cupy array (GPU-compatible) from the NumPy array
#     weights_matrix = cp.zeros((len(word_vocab), 100))
    
#     words_found = 0
#     for i, word in enumerate(word_vocab):
#         try:
#             weights_matrix[i] = cp.array(emb_dict[word])
#             words_found += 1
#         except:
#             pass
    
#     # Transfer the Cupy array to the GPU (if not already on the GPU)
#     weights_matrix = cp.asarray(weights_matrix)
    
#     return weights_matrix, words_found
