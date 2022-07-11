from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os
feature_extractor_model=tf.keras.applications.resnet.ResNet50(input_shape=(160,120,3),include_top=False,weights='imagenet')
feature_extractor_model.trainable=True
def final_model():
  last_layer = feature_extractor_model.get_layer('conv5_block3_out')
  x = (last_layer.output)
  
  #resize=tkl.UpSampling2D(size=(7,7))(inputs)
  x=tf.keras.layers.BatchNormalization()(x)
  x=tf.keras.layers.GlobalMaxPooling2D()(x)
  x=tf.keras.layers.Dense(512,activation='relu',name='RequiredLayer')(x)
  x=tf.keras.layers.BatchNormalization()(x)
  # x=tk.layers.Dense(256,activation='relu')(x)
  # x=tkl.BatchNormalization()(x)
  x=tf.keras.layers.Dense(64,activation='relu')(x)
  x=tf.keras.layers.BatchNormalization()(x)
  x=tf.keras.layers.Dense(10,activation='softmax',name='classification')(x)

  model = tf.keras.Model(feature_extractor_model.input, x)
  my_model = tf.keras.models.clone_model(model)
  return my_model
