# text_classification

## Data set

Assuming you are in the text_classification directory

```
$ cd ../
$ wget http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
$ tar -xvzf 20news-19997.tar.gz
$ mkdir glove.6B
$ cd glove.6B
$ wget https://s3.amazonaws.com/cld2005.glove.100/glove.6B.100d.txt
```


glove:word2vec

GRU:LSTM

CONV:NOCONV


glove:GRU:CONV: 74.54%

glove:GRU:NOCONV: 77.17%

word2vec:GRU:NOCONV: 76.74%

word2vec:GRU:CONV: 74.49%


glove:LSTM:CONV: 71.87%

glove:LSTM:NOCONV 

word2vec:LSTM:NOCONV

word2vec:LSTM:CONV


