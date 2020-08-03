#!/usr/bin/env python
# coding: utf-8

# In[41]:


from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# In[42]:


#reusable print_shape function 
def print_shape(x_train, y_train, x_test, y_test):
    print("x_train:",x_train.shape)
    print("y_train:",y_train.shape)
    print("x_test:",x_test.shape)
    print("y_test:",y_test.shape)


# In[43]:


#download mnist dataset from keraas, split as train, test sets and print all shapes
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print_shape(x_train, y_train, x_test, y_test)


# In[44]:


#https://keras.io/examples/vision/mnist_convnet/ referring to this example
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1) # Make sure images have shape (28, 28, 1)
x_test = np.expand_dims(x_test, -1)
print_shape(x_train, y_train, x_test, y_test)           


# https://stackoverflow.com/questions/20486700/why-we-always-divide-rgb-values-by-255
# 
# RGB (Red, Green, Blue) are 8 bit each.
# The range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
# The combination range is 256*256*256.
# 
# By dividing by 255, the 0-255 range can be described with a 0.0-1.0 range where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF).
# 
# 

# In[45]:


#binary class matrices
no_of_classes = 10 #0-9 digits
y_train = keras.utils.to_categorical(y_train, no_of_classes)
y_test = keras.utils.to_categorical(y_test, no_of_classes)


# In[46]:


#architecture
model = keras.Sequential()

input_shape = (28, 28, 1) #
model.add(keras.Input(shape=input_shape))
#referred to https://keras.io/examples/vision/mnist_convnet/
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()


# In[47]:


#compile, set metrics to accuracy
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[ ]:


#hyper parameters, train
batch_size = 128
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[ ]:


#Evaluations
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1]) 


# In[ ]:




