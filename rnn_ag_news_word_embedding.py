# LSTM with dropout for sequence classification in the IMDB dataset
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding,Flatten
from keras.models import Model
import os
from collections import Counter


def load_data(TEXT_DATA_DIR):
    print('Processing text dataset')
    texts = []  # list of text samples
    labels = []  # list of label ids
    f = open(TEXT_DATA_DIR, 'r')
    for line in f:
        l = line.split('\t')
        labels.append(int(l[0]))
        texts.append(l[1])

    labels_index = (Counter(labels))
    return texts, labels, labels_index


MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100
epochs = 20
batch_size = 64
VALIDATION_SPLIT = 0.2
lat_dim = 256
WORD2VEC = 'WORD2VEC'

EMBED = 'GLOVE'
RNN = 'GRU'
CONV = False
# fix random seed for reproducibility
np.random.seed(7)
BASE_DIR = '../'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
AG_NEWS_TEST = os.path.join('ag_news', 'ag_news_test.txt')
AG_NEWS_TRAIN = os.path.join('ag_news', 'ag_news_train.txt')


def load_glove6B(GLOVE_DIR):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index


def load_word2vec(embeddings_path='embeddings.npz'):
    embeddings_index = {}
    weights = np.load(open(embeddings_path, 'rb'))["emb"]
    layer = Embedding(input_dim=weights.shape[0],
                      output_dim=weights.shape[1],
                      weights=[weights])

    index = np.load(embeddings_path)["word2ind"].flatten()[0]

    for key in index:
        embeddings_index[index[key]] = weights[key]
    return embeddings_index


if EMBED == WORD2VEC:
    embeddings_index = load_word2vec()
    EMBEDDING_DIM = 128
else:
    embeddings_index = load_glove6B(GLOVE_DIR)
    EMBEDDING_DIM = 100
print('Found %s word vectors.' % len(embeddings_index))

texts, labels, labels_index = load_data(AG_NEWS_TRAIN)

print ('labels_index', labels_index)
print ('texts[0]', texts[0])
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
sum = 0
for seq in sequences:
    sum = sum + len(seq)
print ('average len ', sum / len(sequences))

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
labels = to_categorical(np.asarray(labels))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]

x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')
# create the model

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
CONV_STR = 'NOCONV'
if CONV:
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    CONV_STR=CONV

if RNN == 'GRU':
    model.add(GRU(lat_dim, dropout=0.2, recurrent_dropout=0.2))
else:
    model.add(LSTM(lat_dim, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(len(labels_index), activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)  # global max pooling
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# preds = Dense(len(labels_index), activation='softmax')(x)
#
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])

for i in range(5):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    scores = model.evaluate(x_val, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # Final evaluation of the model
    name = "ag_news_rnn_%s_%s_%s_%d_epochs.h5" % (RNN, EMBED, CONV_STR, epochs + i * epochs)
    model.save(name)
