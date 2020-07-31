#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_ function, unicode_literals
import numpy as np [In]
import tensorflow as tf
from tensorflow import keras as ks 
print(tf.__version__) 
#Now, load the Fashion-MNIST data set.
(training_images, training_labels), (test_images, test_ labels) = ks.datasets.fashion_mnist.load_data()


# In[ ]:


print('Training Images Dataset Shape:  {}'. format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'. format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images. shape))
print('No. of Test Images Dataset Labels: {}'. format(len(test_labels)))


# In[ ]:


#As the pixel values range from 0 to 255, we have to rescale these values in the range 0 to 1 before pushing them to the model. We can scale these values (both for training and test data sets) by dividing the values by 255.
training_images = training_images / 255.0 
test_images = test_images / 255.0


# In[ ]:


input_data_shape = (28, 28) 
hidden_activation_function = 'relu' 
output_activation_function = 'softmax' 
dnn_model = models.Sequential()
dnn_model.add(ks.layers.Flatten(input_shape=input_data_ shape, name='Input_layer'))
dnn_model.add(ks.layers.Dense(256, activation=hidden_ activation_function, name='Hidden_layer_1')) 
dnn_model.add(ks.layers.Dense(192, activation=hidden_ activation_function, name='Hidden_layer_2'))
dnn_model.add(ks.layers.Dense(128, activation=hidden_ activation_function, name='Hidden_layer_3'))
dnn_model.add(ks.layers.Dense(10, activation=output_ activation_function, name='Output_layer')) 
dnn_model.summary() 


# In[ ]:


optimizer = 'adam' 
loss_function = 'sparse_categorical_crossentropy' metric = ['accuracy']
dnn_model.compile(optimizer=optimizer, loss=loss_ function, metrics=metric)
dnn_model.fit(training_images, training_labels, epochs=20) 


# In[ ]:


#Training valuation
training_loss, training_accuracy = dnn_model. evaluate(training_images, training_labels)
print('Training Data Accuracy {}'. format(round(float(training_accuracy),2)))


# In[ ]:


#Test evaluation
test_loss, test_accuracy = dnn_model.evaluate(test_ images, test_labels)
print('Test Data Accuracy {}'.format(round(float(test_ accuracy),2))) 

