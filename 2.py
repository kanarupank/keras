#!/usr/bin/env python
# coding: utf-8

# In[67]:


from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


# In[68]:


#reusable print_shape function 
def print_shape(x_train, y_train, x_test, y_test):
    print("x_train:",x_train.shape)
    print("y_train:",y_train.shape)
    print("x_test:",x_test.shape)
    print("y_test:",y_test.shape)


# In[69]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print_shape(x_train, y_train, x_test, y_test)


# In[70]:


noise_factor=0.5 #tried for 0.25, 0.4, 0.5
x_train_noicy=x_train+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_train.shape)
x_test_noicy=x_test+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_test.shape)
x_train_noicy=np.clip(x_train_noicy,0.,1.)
x_test_noicy=np.clip(x_test_noicy,0.,1.)


# In[71]:


x_train_noicy = np.expand_dims(x_train_noicy, -1) # Make sure images have shape (28, 28, 1)
x_test_noicy = np.expand_dims(x_test_noicy, -1)


# In[72]:


no_of_classes = 10 #digits 0-9
y_train = keras.utils.to_categorical(y_train, no_of_classes)
y_test = keras.utils.to_categorical(y_test, no_of_classes)


# In[73]:


model = keras.Sequential()
input_shape = (28, 28, 1)
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(no_of_classes, activation="softmax"))
model.summary()


# In[74]:


#hyper params
batch_size = 128
epochs = 10
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train_noicy, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[75]:


#evalutions
score = model.evaluate(x_test_noicy, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

