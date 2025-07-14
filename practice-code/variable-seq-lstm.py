import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
import numpy as np

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras import initializers
# from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.callbacks import Callback
class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/pdf/1512.08756.pdf]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.step_dim = input_shape[1]
        assert len(input_shape) == 3  # batch, timestep, num_features
        print(input_shape)
        self.W = self.add_weight((input_shape[-1],),  # num_features
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),  # timesteps
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        # print(K.reshape(x, (-1, features_dim)))  # n, d
        # print(K.reshape(self.W, (features_dim, 1)))  # w= dx1
        # print(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))))  # nx1

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # batch, step
        # print(eij)
        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # print(a)
        a = K.expand_dims(a)
        # print("expand_dims:")
        # print(a)
        # print("x:")
        # print(x)
        weighted_input = x * a
        # print(weighted_input.shape)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim



model = Sequential()

model.add(LSTM(256, return_sequences=True, input_shape=(None, 256), batch_input_shape=(32, None, 256)))
model.add(Attention(256))
model.add(Dense(256, activation='softmax'))


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[auc])

print(model.summary(90))


def train_generator():
    while True:
        sequence_length =
        x_train =
        # y_train will depend on past 5 timesteps of x
        y_train =
        for i in range(1, 5):
            y_train[:, i:] += x_train[:, :-i, i]
        y_train = to_categorical(y_train > 2.5)
        yield x_train, y_train

model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)





with K.get_session():

    # create model
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, stateful=True, input_shape=(None, 256),
             batch_input_shape=(32, None, 256)))
    model.add(LSTM(256))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['auc'])


    score = model.evaluate(x_val, y_val, batch_size=32, verbose=1)`