'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding,Flatten
from keras.models import Model


BASE_DIR = '../'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

# print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# # second, prepare text samples and their labels
# print('Processing text dataset')

texts = []  # list of text samples
labels = []  # list of label ids
f = open('ag_news/ag_news_train.txt','r')
for line in f:
    l = line.split('\t')
    labels.append(int(l[0]))
    texts.append(l[1])
texts_val = []  # list of text samples
labels_val = []  # list of label ids

#f = open('ag_news/ag_news_test.txt','r')
#for line in f:
#    l = line.split('\t')
#    labels_val.append(int(l[0]))
#    texts_val.append(l[1])
# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 if sys.version_info < (3,):
#                     f = open(fpath)
#                 else:
#                     f = open(fpath, encoding='latin-1')
#                 t = f.read()
#                 i = t.find('\n\n')  # skip header
#                 if 0 < i:
#                     t = t[i:]
#                 texts.append(t)
#                 f.close()
#                 labels.append(label_id)

print ('text')
print (texts[0])
print ('label')
print (labels[0])

from collections import Counter
labels_index = (Counter(labels))

print ('label index')
print (labels_index)
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# print('sequence'+str(len(sequences)))
# print(sequences[0])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print (data.shape)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# x_train = data
# y_train = labels
# x_val = data_val
# y_val = labels

print('Preparing embedding matrix.')

# prepare embedding matrix

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
# no embeddings
# num_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i >= MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

# no embeddings
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             input_length=MAX_SEQUENCE_LENGTH)
print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))