# summary     Sentiment Analysis
# description Sentiment Analysis for Image
# version     1.0.3
# file        retrain.py
# author      2KHA 
# contact     tla.atc.co.nl@gmail.com
# copyright   Copyright (C) Mango AI
#  
# This source file is free software, available under the following license:
#    GNU GENERAL PUBLIC LICENSE license - https://github.com/2kha/Visualizations/blob/main/LICENSE
# 
# This source file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the license files for details.
# 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from keras import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.optimizers import Adam
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint

base_dir = 'D:\\Projects\\V3A\\Datasets'
items = '\\Items\\'
features = '\\Features\\'
images = '\\Images\\'
records = '\\Records\\'

epochs = 5000
batch_size = 32
iterations = 120     # 192 * 20 / 32
weight_decay = 0.01

# load feature vectors and labels
x_train = np.genfromtxt(base_dir + features + 'train_features.csv', dtype=np.float32, delimiter=',')
x_val   = np.genfromtxt(base_dir + features + 'val_features.csv', dtype=np.float32, delimiter=',')
y_train = np.genfromtxt(base_dir + features + 'train_labels.csv', dtype=np.float32, delimiter=',')
y_val   = np.genfromtxt(base_dir + features + 'val_labels.csv', dtype=np.float32, delimiter=',')


def process_single_sample(img, label):
    return {"image": img, "label": label}

def get_train_dataset():

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_dataset = (
        train_dataset.map(
            process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
       
    return train_dataset

def get_validation_dataset():

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    validation_dataset = (
        validation_dataset.map(
            process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
  
    return validation_dataset

train_data = get_train_dataset()
val_data = get_validation_dataset()

class SIMPLE_LOSS(layers.Layer):

    def __init__(self, name=None):

        super().__init__(name=name)       

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        
        valance_pre = tf.reshape(y_pred[:, 0], [-1])
        arousal_pre =  tf.reshape(y_pred[:, 1], [-1])
        affinity_pre =  tf.reshape(y_pred[:, 2], [-1])
        accrual_pre =  tf.reshape(y_pred[:, 3], [-1])

        valance_gt = tf.reshape(y_true[:, 0], [-1])
        arousal_gt= tf.reshape(y_true[:, 1], [-1])
        affinity_gt = tf.reshape(y_true[:, 2], [-1])
        accrual_gt= tf.reshape(y_true[:, 3], [-1])

        valance_loss = tf.norm(tf.subtract(valance_pre, valance_gt))
        arousal_loss= tf.norm(tf.subtract(arousal_pre, arousal_gt))
        affinity_loss = tf.norm(tf.subtract(affinity_pre, affinity_gt))
        accrual_loss= tf.norm(tf.subtract(accrual_pre, accrual_gt))

        va_loss = tf.add(valance_loss, arousal_loss)
        aa_loss = tf.add(affinity_loss, accrual_loss)

        loss = tf.reduce_mean(tf.reduce_sum(tf.add(va_loss, aa_loss )))

        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model():

    # Input with shape of height=32 and width=128 
    inputs = Input(shape=(2048), name="image")
    labels = Input(shape=(4), name="label")

    dense1 = Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x1 = Dense(1, kernel_regularizer=regularizers.l2(weight_decay))(dense1)

    dense2 = Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x2 = Dense(1,  kernel_regularizer=regularizers.l2(weight_decay))(dense2)

    dense3 = Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x3 = Dense(1,  kernel_regularizer=regularizers.l2(weight_decay))(dense3)

    dense4 = Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x4 = Dense(1,  kernel_regularizer=regularizers.l2(weight_decay))(dense4)

    y = layers.Concatenate()([x1, x2, x3, x4])

    output = SIMPLE_LOSS(name="loss")(labels, y) 

    model = Model(inputs=[inputs, labels], outputs=output)

    model.compile(optimizer=Adam())

    return model


model = build_model()

model.summary

# set up callback
cur_time = str(int(time.time()))

cbks = [
    TensorBoard(log_dir= base_dir + features +'sentiment_' + cur_time),
    ModelCheckpoint(base_dir + features + 'ckpt\\' + cur_time + '_{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
]

# training
history = model.fit(
    train_data,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=False,      # already shuffled during augmentation
    validation_data=val_data,
    callbacks=cbks,
    verbose=1
)

# save and plot result
model.save(base_dir + features + 'sentiment_model_{}.h5'.format(cur_time))

train_error = [(1-acc)*100 for acc in history.history['loss']]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.tight_layout(pad=3, w_pad=2)
fig.suptitle('Sentiment Classifier', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Error(%)', fontsize=14)
ax1.plot(train_error, label='Training Error')
ax1.legend()

ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.legend()

plt.savefig(base_dir + features + 'sentiment_model_{}.png'.format(cur_time))

