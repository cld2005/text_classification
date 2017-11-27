import numpy as np
from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model


def word2vec_embedding_layer(embeddings_path='../../Downloads/embeddings.npz'):
    embeddings_index={}
    weights = np.load(open(embeddings_path, 'rb'))["emb"]
    layer = Embedding(input_dim=weights.shape[0],
                      output_dim=weights.shape[1],
                      weights=[weights])

    index = np.load(embeddings_path)["word2ind"].flatten()[0]

    for key in index:
        embeddings_index[index[key]]= weights[key]
    return embeddings_index


if __name__ == '__main__':
    embeddings_index = word2vec_embedding_layer()
    print (embeddings_index['the'])
    print (embeddings_index['UNK'])
    print (embeddings_index['in'])