#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# In[6]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


# In[8]:


no_of_classes = 10
input_shape = (28, 28, 1)

noise_factor=.5
x_train_noise=x_train+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_train.shape)
x_test_noise=x_test+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_test.shape)
x_train_noise=np.clip(x_train_noise,0.,1.)
x_test_noise=np.clip(x_test_noise,0.,1.)

x_train_noise = np.expand_dims(x_train_noise, -1) # Make sure images have shape (28, 28, 1)
x_test_noise = np.expand_dims(x_test_noise, -1)

y_train = keras.utils.to_categorical(y_train, no_of_classes)
y_test = keras.utils.to_categorical(y_test, no_of_classes)


# In[9]:


#add encoder
Input_img = layers.Input(shape=(28, 28, 1))  
x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(Input_img)
x2 = layers.MaxPooling2D((2, 2), padding='same')(x1)
x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x3)

#add decoder
x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x2 = layers.UpSampling2D((2, 2))(x3)
x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
x = layers.UpSampling2D((2, 2))(x1)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer="adam", loss='binary_crossentropy')


# In[10]:


autoencoder.fit(x_train_noise, x_train,
                epochs=20,
                batch_size=128)


# In[13]:


#predict
predicted=autoencoder.predict(x_test_noise)

#denoised input
train_renoised = autoencoder.predict(x_train_noise)
test_renoised =autoencoder.predict(x_test_noise)

#build classifer model
model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(no_of_classes, activation="softmax"))
model.summary()


# In[14]:


batch_size = 128
epochs = 10
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_renoised, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[15]:


#Evaluate
score = model.evaluate(test_renoised, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
#observing a lower accuracy might need to try more epochs or with other configurations


# In[ ]:




