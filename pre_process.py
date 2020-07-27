import csv
import json
import re

from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def read_word_embedding(embedding_path, word_embedding_dim):
    embeddings, word_id = dict(), dict()
    embeddings[0] = np.zeros(word_embedding_dim, dtype="float32")  # 填充
    word_id["<UNK>"] = 0
    with open(embedding_path, "r", encoding="utf-8") as f:
        ids = 1
        for line in f:
            values = np.array(line.strip().split())
            if len(values) - 1 == word_embedding_dim:
                embeddings[ids] = np.array(values[1:], dtype="float32")
                word_id[values[0]] = ids
                ids += 1
    return embeddings, word_id  # embeddings{id: embedding},word_id: {word: id}


def get_embedding_matrix(embeddings, word_embedding_dim):
    embedding_matrix = np.zeros((len(embeddings), word_embedding_dim))
    for i, vec in embeddings.items():
        embedding_matrix[i] = vec
    return embedding_matrix


def word_to_ids(word_sequence, word_id):
    words_id_sequence = []
    for word in word_sequence:
        if word in word_id:
            words_id_sequence.append(word_id[word])
        elif word.lower() in word_id:
            words_id_sequence.append(word_id[word.lower()])
        else:
            words_id_sequence.append(0)
    return words_id_sequence


def review_sentences(text, min_char=10, pattern=r"[\n.!\?;]"):
    return [sent.strip() for sent in re.split(pattern, text) if len(sent) > min_char]


def sentence_words(sent):
    return WordPunctTokenizer().tokenize(sent)


def read_yelp_json(filepath, default_star=3):
    RUs, RIs = dict(), dict()
    with open(filepath, encoding='utf-8') as f:
        for line in f.readlines():
            review = json.loads(line)
            RUs.setdefault(review['user_id'], list()).append([review['business_id'], review['text'], review['stars']])
            RIs.setdefault(review['business_id'], list()).append([review['user_id'], review['text']])

    dataset = []
    for uid, rus in RUs.items():
        for bid, ris in RIs.items():
            RUI, RU, RI = list(), list(), list()
            yUI, y_count = 0, 0
            for ru in rus:
                if ru[0] == bid:  # 分句
                    RUI.extend(review_sentences(ru[1]))
                    yUI += ru[2]  # rating sum
                    y_count += 1
                else:
                    RU.extend(review_sentences(ru[1]))
            for ri in ris:
                if ri[0] != uid:
                    RI.extend(review_sentences(ri[1]))
            dataset.append([RUI, RU, RI, float(yUI / y_count if y_count > 0 else default_star), uid, bid])
    return np.array(dataset)  # shape=(样本个数, RUI[] RU[] RI[] yUI uid bid, sentence)


def handle_sent_list(R_sent, word_id, sent_length, sequence_length):
    Rs = [word_to_ids(sentence_words(sent), word_id) for sent in R_sent]  # 传入句子列表，分词->映射
    R = []
    for sent_list in pad_sequences(Rs, maxlen=sent_length, padding='post'):  # 统一句长->连接
        R.extend(sent_list)
    return pad_sequences([R], maxlen=sequence_length, padding='post', truncating='post')[0]  # 统一总评论长度


def get_training_data(dataset, word_id, sent_length, sequence_length):
    RUIs, RUs, RIs, yUIs = list(), list(), list(), list()
    for sample in dataset:
        RUI = handle_sent_list(sample[0], word_id, sent_length, sequence_length)
        RU = handle_sent_list(sample[1], word_id, sent_length, sequence_length)
        RI = handle_sent_list(sample[2], word_id, sent_length, sequence_length)
        RUIs.append(RUI)
        RUs.append(RU)
        RIs.append(RI)
        yUIs.append(sample[3])
    return np.array(RUIs), np.array(RUs), np.array(RIs), np.array(yUIs)


# if __name__ == '__main__':
#     train_path = "./data/reviews_small.json"
#     embedding_path = "embedding/glove.twitter.27B.50d.txt"
#     word_embedding_dim = 50  # set according to embedding_path
#
#     print("###### Load word embedding! ######")
#     embeddings, word_id = read_word_embedding(embedding_path, word_embedding_dim)
#     # embedding_matrix = get_embedding_matrix(embeddings)
#
#     print("###### Reading data! ######")
#     dataset = read_yelp_json(train_path)
#     training_data = get_training_data(dataset, word_id, 50, 500)
#
#     print("###### display data sample! ######")
#     sample_id = 2
#     print("a sample of dataset[%d]:" % sample_id)
#     print('user_id:', dataset[sample_id][4], 'business_id:', dataset[sample_id][5])
#     for i in range(3):
#         for samp in dataset[sample_id][i]:
#             print(samp)
#         print("--------------------------------------")
#     print("a sample of training data[%d]:" % sample_id)
#     print("RUI:", training_data[sample_id][0])
#     print("RU :", training_data[sample_id][1])
#     print("RI :", training_data[sample_id][2])
