import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, LSTM, \
    Flatten, Embedding, Multiply, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py


def vqa_model(embedding_matrix,
              num_words,
              embedding_dim,
              seq_length,
              dropout_rate,
              num_classes
              ):
    ###########################
    # Word2Vec Model
    ###########################
    print("Creating text model ...")
    input_txt = Input(shape=(seq_length,), name='text_input')
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix] ,trainable=False)(input_txt)
    x = LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim))(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=512, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    output_txt = Dense(1024, activation='tanh')(x)
    txt_model = Model(input_txt, output_txt)
    txt_model.summary()

    ###########################
    # Image Model
    ###########################
    print("Creating image model ...")
    input_img = Input(shape=(4096,), name='image_input')
    output_img = Dense(1024, activation='tanh')(input_img)
    img_model = Model(input_img, output_img)
    img_model.summary()

    ###########################
    # VQA Model
    ###########################
    print("Creating vqa model...")
    input_intermidiate_img = Input(shape=(1024,), name='intermidiate_image_input')
    input_intermidiate_txt = Input(shape=(1024,), name='input_intermidiate_txt_input')
    x = Multiply()([input_intermidiate_img, input_intermidiate_txt])
    x = Dropout(dropout_rate)(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    vqa = Dense(num_classes, activation='softmax')(x)
    vqa_model = Model([input_intermidiate_img, input_intermidiate_txt], vqa)
    vqa_model.summary()
    # internal connection
    output_vqa = vqa_model([img_model(input_img), txt_model(input_txt)])

    ###########################
    # VQA Model
    ###########################
    print("Packing multi model...")
    multiModel = Model([input_img, input_txt], output_vqa, name='multiModel')
    multiModel.summary()

    # optimizer
    multiModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return multiModel
