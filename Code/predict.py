# summary     Sentiment Analysis
# description Sentiment Analysis for Image
# version     1.0.3
# file        predict.py
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
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import keras

from keras import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.optimizers import Adam
from keras import layers
from keras.applications.xception import Xception
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from glob import glob


base_dir = 'D:\\Projects\\V3A\\Datasets'
items = '\\Items\\'
features = '\\Features\\'
images = '\\Images\\'
records = '\\Records\\'

model_name = 'sentiment_model_1768082008.h5'

model_path = base_dir + features + model_name

test_path = "D:\\Projects\\AI\\Web Management\\AI\\ModService\\Upload"

labels = [
             "None"			    #	1
             ,"Addicted"		#	2
             ,"Amused"			#	3
             ,"Angry"			#	4
             ,"Annoyed"			#	5
             ,"Anxious"			#	6
             ,"Bored"			#	7
             ,"Calm"			#	8
             ,"Cheerful"		#	9             
             ,"Compassionate"	#	10
             ,"Confident"		#	11
             ,"Contented"		#	12
             ,"Couragous"		#	13
             ,"Craving"			#	14
             ,"Curious"			#	15
             ,"Delighted"		#	16
             ,"Depressed"		#	17
             ,"Diligent"		#	18
             ,"Dislike"			#	19
             ,"Disgusting"		#	20
             ,"Doubtful"		#	21
             ,"Eager"			#	22
             ,"Ecstatic"		#	23
             ,"Enchanted"		#	24
             ,"Enthralled"		#	25
             ,"Envious"			#	26
             ,"Excited"			#	27
             ,"Fear"			#	28
             ,"Generous"		#	29
             ,"Greedy"			#	30
             ,"Guilty"			#	31
             ,"Happy"			#	32
             ,"Hatred"			#	33
             ,"Hopeful"			#	34
             ,"Hurt"			#	35
             ,"Indulgent"		#	36
             ,"Jealous"         #	37
             ,"Like"			#	38
             ,"Love"			#	39
             ,"Lustful"			#	40
             ,"Needy"			#	41
             ,"Nostaligic"		#	42
             ,"Passionate"		#	43
             ,"Pleased"			#	44
             ,"Proud"			#	45
             ,"Sad"				#	46
             ,"Shameful"		#	47
             ,"Statisfied"		#	48
             ,"Suprised"		#	49
             ,"Tempted"			#	50
             ,"Touched"			#	51
             ,"Trust"			#	52
             ,"Uninteresed"		#	53
             ,"Missing"			#	54
    ]


# sentiments are calculated as anchor points in sentiment space, known as V3A space
categories = [
    
     [ 0.0, 0.0, 0.0, 0.0          ]		#	1
    ,[ 0.9 , 0.9 , 0.9 , -0.8      ]		#	2
    ,[ 0.8 , 0.7 , 0.5 , 0.1       ]		#	3
    ,[ -0.8 , 0.8 , -0.8 , -0.2    ]		#	4
    ,[ -0.3 , 0.3 , -0.3 , -0.01   ]		#	5
    ,[ -0.3 , 0.4 , -0.1 , -0.5    ]		#	6
    ,[ 0.2 , 0.3 , 0.2 , 0         ]		#	7
    ,[ 0.1 , 0 , 0.1 , 0.01        ]		#	8
    ,[ 0.5 , 0.5 , 0.1 , 0.1       ]		#	9
    ,[ 0.6 , 0.4 , 0.5 , -0.1      ]		#	10
    ,[ 0.7 , 0.4 , 0.1 , 0.7       ]		#	11
    ,[ 0.3 , 0.1 , 0 , 0           ]		#	12
    ,[ 0.7 , 0.1 , -0.7 , -0.5     ]		#	13
    ,[ 0.3 , 0.9 , 0.9 , 0.9       ]		#	14
    ,[ 0.3 , 0.6 , 0.1 , 0.1       ]		#	15
    ,[ 0.8 , 0.5 , 0 , 0           ]		#	16
    ,[ -0.8 , 0.6 , 0 , -0.7       ]		#	17
    ,[ 0.4 , 0.2 , 0 , -0.3        ]		#	18
    ,[ -0.2 , 0.2 , -0.2 , -0.1    ]		#	19
    ,[ -0.9 , 0.7 , -0.9 , 0       ]		#	20
    ,[ -0.4 , 0.3 , -0.5 , -0.1    ]		#	21
    ,[ 0 , 0.7 , 0.01 , 0          ]		#	22
    ,[ 0.8 , 0.8 , 0.7 , 0.7       ]		#	23
    ,[ 0.5 , 0.7 , 0.5 , -0.3      ]		#	24
    ,[ 0.4 , 0.5 , 0.4 , -0.1      ]		#	25
    ,[ -0.5 , 0.5 , -0.2 , 0.5     ]		#	26
    ,[ 0.1 , 0.9 , 0.01 , 0        ]		#	27
    ,[ -0.7 , 0.7 , -0.2 , -0.8    ]		#	28
    ,[ 0.8 , 0.2 , 0.7 , -0.5      ]		#	29
    ,[ -0.2 , 0.7 , -0.3 , 0.7     ]		#	30
    ,[ -0.7 , 0.5 , -0.1 , -0.6    ]		#	31
    ,[ 0.7 , 0.6 , 0 , 0           ]		#	32
    ,[ -0.8 , 0.8 , -0.9 , -0.9    ]		#	33
    ,[ 0.1 , 0.1 , 0.2 , 0.1       ]		#	34
    ,[ -0.9 , 0.5 , 0 , -0.1       ]		#	35
    ,[ 0.6 , 0.5 , 0.5 , -0.5      ]		#	36
    ,[ -0.4 , 0.6 , 0.6 , -0.7     ]		#	37
    ,[ 0.3 , 0.5 , 0.4 , 0         ]        #	38
    ,[ 0.7 , 0.8 , 0.8 , 0.01      ]	    #	39
    ,[ 0.7 , 0.8 , 0.4 , 0.7       ]		#	40
    ,[ -0.3 , 0.4 , 0.3 , 0.4      ]		#	41
    ,[ 0.6 , 0.5 , 0.6 , -0.7      ]		#	42
    ,[ 0.5 , 0.8 , 0.3 , 0.6       ]		#	43
    ,[ 0.3 , 0.3 , 0 , 0           ]		#	44
    ,[ 0.5 , 0.3 , -0.5 , 0.7      ]		#	45
    ,[ -0.5 , 0.4 , 0 , -0.1       ]		#	46
    ,[ -0.5 , 0.4 , -0.1 , -0.7    ]		#	47
    ,[ 0.5 , 0.3 , 0.1 , 0.2       ]		#	48
    ,[ 0.1 , 0.85 , -0.1 , -0.1    ]		#	49
    ,[ 0.6 , 0.6 , 0.7 , -0.2      ]		#	50
    ,[ 0.7 , 0.4 , 0.2 , 0.1       ]		#	51
    ,[ 0.1 , 0.2 , 0.7 , 0         ]		#	52
    ,[ 0.01 , 0.001 , 0 , 0        ]		#	53
    ,[ -0.2 , 0.3 , 0.5 , -0.5     ]		#	54
    ]


img_size = 299  # match Xception input size

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

    dense1 = Dense(512, activation='tanh')(inputs)
    x1 = Dense(1)(dense1)

    dense2 = Dense(512, activation='tanh')(inputs)
    x2 = Dense(1)(dense2)

    dense3 = Dense(512, activation='tanh')(inputs)
    x3 = Dense(1)(dense3)

    dense4 = Dense(512, activation='tanh')(inputs)
    x4 = Dense(1)(dense4)

    y = layers.Concatenate()([x1, x2, x3, x4])

    output = SIMPLE_LOSS(name="loss")(labels, y) 

    model = Model(inputs=[inputs, labels], outputs=output)

    model.compile(optimizer=Adam())

    return model

def resize(image):
    h, w, c = image.shape
    cropped = image

    h, w, c = cropped.shape
    if h > img_size:    # shrink
        return cv.resize(cropped, (img_size, img_size), interpolation=cv.INTER_AREA)
    elif h < img_size:  # enlarge
        return cv.resize(cropped, (img_size, img_size), interpolation=cv.INTER_CUBIC)
    else:
        return cropped

def load_test_images(file_list):
    
    test_set = list()
    test_set_rgb = list()
        
    for file in file_list:
        print(file)
        img = cv.imread(file)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = resize(img)
        img_rgb = resize(img_rgb)            
        test_set.append(img)
        test_set_rgb.append(img_rgb)

    return np.asarray(test_set), np.asarray(test_set_rgb)


# load test images
test_dir = test_path
filenames = glob(os.path.join(test_dir, '*.*'))
images, images_rgb = load_test_images(filenames)

# calculate from the training set
channel_mean = np.array([110.73151039, 122.90935242, 136.82249855])
channel_std = np.array([69.39734207, 67.48444001, 66.66808662])

# normalize images
images = images.astype('float32')
for j in range(3):
    images[:, :, :, j] = (images[:, :, :, j] - channel_mean[j]) / channel_std[j]

# make predictions
base_model = Xception(include_top=False, weights='imagenet', pooling='avg')

sentiment_model = build_model()

sentiment_model.load_weights(model_path)

prediction_model = keras.models.Model(
        sentiment_model.get_layer(name="image").input, layers.Concatenate()([sentiment_model.get_layer(index=5).output, sentiment_model.get_layer(index=6).output, sentiment_model.get_layer(index=7).output, sentiment_model.get_layer(index=8).output]) 
)


features = base_model.predict(images)
predictions = prediction_model.predict(features)

values = []

for i in range(len(categories)):
    values.append([x + 1 for x in categories[i]])

for i in range(len(images_rgb)):
    # find the index of the label with the smallest corresponding
    # distance, then extract the distance and label
    
    label = "I am feeling:"
    prediction = predictions[i]

    sentiment = [x - 1 for x in prediction]
  
    distances = np.linalg.norm(values - prediction, axis=1)

    norms = np.transpose(np.vstack((distances, labels)))

    results =  sorted(norms, key=lambda x: x[0])

    description = ",".join([str(x[1]) for x in results[0:5]])

    # output
    print("Output,{}\t\t{}".format(label + description,   filenames[i]))


input()

