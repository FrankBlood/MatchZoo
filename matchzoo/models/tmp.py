import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Input, Dense, Conv2D, Lambda, Flatten, Reshape
from keras.models import Model


def load_glove(path):
    """
    creates a dictionary mapping words to vectors from a file in glove format.
    """
    with open(path) as f:
        glove = {}
        for line in f.readlines():
            values = line.strip().split(" ")
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove[word] = vector
        return glove


def cal_match_matrix(query, doc, max_len_1, max_len_2, w2v, embed_size):
    """
    """
    qterms = query.split()
    dterms = doc.split()

    # embedding matrix for query
    qv = []
    dlen_1 = max_len_1 - len(qterms)
    if dlen_1 > 0:
        for t in qterms:
            qv.append(w2v[t])

        for i in range(0, dlen_1):
            qv.append(np.zeros(embed_size))
    else:
        for i in range(0, dlen_1):
            t = qterms[i]
            qv.append(w2v[t])
    qv = np.array(qv)

    # embedding matrix for doc
    dv = []
    dlen_2 = max_len_2 - len(dterms)
    if dlen_2 > 0:
        for t in dterms:
            dv.append(w2v[t])

        for i in range(0, dlen_2):
            dv.append(np.zeros(embed_size))
    else:
        for i in range(0, dlen_2):
            t = dterms[i]
            dv.append(w2v[t])
    dv = np.array(dv)

    # mm = np.dot(qv, dv.T)
    mm = cosine_similarity(qv, dv)

    return mm


def cal_binsum(mm, bin_num=20):
    mbinsum = np.zeros(bin_num, dtype=np.float32)
    for (i, j), v in np.ndenumerate(mm):
        vid = int((v + 1.) / 2. * (bin_num - 1.))
        mbinsum[vid] += v
    return mbinsum.flatten()

query = "much 1985 coin worth"
doc = "how much is the queen elizabeth 1985 coin worth"
max_len_1 = 10
max_len_2 = 10
embed_size = 50

# mm = cal_match_matrix(query, doc, max_len_1, max_len_2, glove, embed_size)

def test(mm, bin_num=20):
    a = 0
    for i, j in mm[:, 1]:
        a += (i+j)
    return a

inputs = Input(shape=(1, 10, 10))

conv2d = Conv2D(filters=5, kernel_size=3, data_format='channels_first', padding='valid', activation='relu')(inputs)

print(conv2d.get_shape())
# fconv2d = Flatten()(conv2d)
# print(fconv2d.get_shape())

fconv2d = Reshape((2, -1))(conv2d)
print(fconv2d.get_shape())

# a = Lambda(lambda x: test(x))(conv2d)
# print(a.get_shape())