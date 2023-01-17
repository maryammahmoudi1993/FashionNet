import tensorflow as tf  
from keras import layers, models
from main import load_data

path = "D:\Python_Codes\Deep_Learning\Advanced Tensorflow\Multi_Label Classification\FashionNet\Week2_dataset\clothes_dataset\dataset"
load_data(path=path)    


class FashionNet():
    
    def build(): # VGG16 architecture
        input_layer = layers.Input(shape=())