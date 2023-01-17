import tensorflow as tf  
from keras import layers, models

class FashionNet():
    
    def build(): # VGG16 architecture
        input_layer = layers.Input(shape=())