import os
import tarfile
import gzip
import shutil
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Activation, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
%matplotlib inline


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


DS_DOWNLOAD_URL = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
GLOVE_DOWNLOAD_URL = 'https://github.com/kmr0877/IMDB-Sentiment-Classification-CBOW-Model/raw/master/glove.6B.50d.txt.gz'
DOWNLOAD_LOC = 'input/'
TRAIN_DIR_NAME = '20news-bydate-train'
TEST_DIR_NAME = '20news-bydate-test'
TRAIN_INPUT_DIR = os.path.join(DOWNLOAD_LOC + TRAIN_DIR_NAME)
TEST_INPUT_DIR = os.path.join(DOWNLOAD_LOC + TEST_DIR_NAME)
GLOVE_FILE_NAME = 'glove.6B.50d.txt'
GLOVE_FILE_PATH = os.path.join(DOWNLOAD_LOC + GLOVE_FILE_NAME)
SAVE_MODEL_PATH = 'output/20news_model.h5'

MAX_DOC_LEN = 2000
MAX_NUM_WORDS = 20000


def fetch_extract(download_url, download_loc, comp_name, comp_type='tgz', decomp_name=None):
    os.makedirs(download_loc, exist_ok=True)
    comp_path = os.path.join(download_loc, comp_name)
    urllib.request.urlretrieve(download_url, comp_path)
    if comp_type == 'tgz':
        with tarfile.open(comp_path) as tgz_in:
            tgz_in.extractall(path=download_loc)
    elif comp_type == 'gz':
        decomp_path = os.path.join(download_loc, decomp_name)
        with gzip.open(comp_path, 'rb') as gz_in:
            with open(decomp_path, 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)
    os.remove(comp_path)


def get_classes(classes_dir):
    # each folder contains documents about the same topic. We'll use folder names as classes
    class_names = {name for name in os.listdir(classes_dir) if os.path.isdir(os.path.join(classes_dir, name))}
    # map class to indexes and vice versa
    class_to_idx = {}
    idx_to_class = {}
    for i, c in enumerate(class_names):
        class_to_idx[c] = i
        idx_to_class[i] = c
    return class_to_idx, idx_to_class

# read training examples, example identifiers (file names), and classes (directories)


def read_train(class_to_idx):
    train_texts = []  # training text corpuses
    train_file_names = []  # training examples' ids
    train_classes = []  # training examples' classes
    # read training files
    for dirname, _, filenames in os.walk(TRAIN_INPUT_DIR):
        for filename in filenames:
            train_file_names.append(filename)
            with open(os.path.join(dirname, filename), 'r', encoding='latin-1') as text_file:
                train_texts.append(text_file.read())
            dirname_tokens = dirname.split('/')
            train_classes.append(dirname_tokens[-1])
    # transform classes from text to integral indices
    train_y = list(map(lambda cl: class_to_idx[cl], train_classes))
    # encode the class indices into one-hot vectors
    train_y = to_categorical(train_y)
    return train_texts, train_file_names, train_y

# read validuation examples, example identifiers (file names), and classes (directories)


def read_test(class_to_idx):
    valid_texts = []
    valid_file_names = []
    valid_classes = []
    for dirname, _, filenames in os.walk(TEST_INPUT_DIR):
        for filename in filenames:
            valid_file_names.append(filename)
            with open(os.path.join(dirname, filename), 'r', encoding='latin-1') as text_file:
                valid_texts.append(text_file.read())
            dirname_tokens = dirname.split('/')
            valid_classes.append(dirname_tokens[-1])
    valid_y = list(map(lambda cl: class_to_idx[cl], valid_classes))
    valid_y = to_categorical(valid_y)
    return valid_texts, valid_file_names, valid_y

# read and parse the GloVe file


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map

# replace each word of a sentence with its index in the vocabulary


def sentences_to_indices(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences=sequences,
                         maxlen=MAX_DOC_LEN, padding='post', truncating='post')
    return data, word_index


def create_embedding_layer(word_to_vec_map, word_to_index):
    # Keras requires adding 1 to the vocabulary length
    vocab_len = min(MAX_NUM_WORDS, len(word_to_index) + 1)
    emb_dim = len(word_to_vec_map["the"])
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, idx in word_to_index.items():
        if idx >= MAX_NUM_WORDS:
            continue
        emb_vector = word_to_vec_map.get(word)
        if emb_vector is not None:
            emb_matrix[idx] = emb_vector
    embedding_layer = Embedding(
        vocab_len, emb_dim, input_length=MAX_DOC_LEN, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def create_model_rnn(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = create_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    x = LSTM(units=128, dropout=0.5, return_sequences=True)(embeddings)
    x = LSTM(units=128, dropout=0.5)(x)
    x = Dense(units=20)(x)
    x = Activation(activation='softmax')(x)
    model = Model(inputs=sentence_indices, outputs=x)
    return model


def create_model_cnn(input_shape, word_to_vec_map, word_to_index, num_classes):
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = create_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    x = Conv1D(128, 5, activation='relu')(embeddings)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=sentence_indices, outputs=preds)
    return model


if __name__ == "__main__":
    fetch_extract(DS_DOWNLOAD_URL, DOWNLOAD_LOC, '20news-bydate.tar.gz')
    fetch_extract(GLOVE_DOWNLOAD_URL, DOWNLOAD_LOC, 'glove.6B.50d.txt.gz', comp_type='gz', decomp_name=GLOVE_FILE_NAME)
    class_to_idx, idx_to_class = get_classes(TRAIN_INPUT_DIR)
    train_texts, train_file_names, train_y = read_train(class_to_idx)
    test_texts, test_file_names, valid_y = read_test(class_to_idx)
    texts_indices, word_to_index = sentences_to_indices(train_texts + valid_texts)
    word_to_vec_map = read_glove_vecs(GLOVE_FILE_PATH)
    model = create_model_rnn((MAX_DOC_LEN,), word_to_vec_map, word_to_index, len(class_to_idx))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.fit(x=texts_indices[:len(train_texts)], y=train_y, epochs=10, batch_size=32,
              shuffle=True, validation_data=(texts_indices[len(train_texts):], valid_y))
    model.save(SAVE_MODEL_PATH)
