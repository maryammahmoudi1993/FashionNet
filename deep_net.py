import tensorflow as tf  
from keras import layers, models
from data_preprocessing import PreProcessing

path = "D:\Python_Codes\Deep_Learning\Advanced Tensorflow\Multi_Label Classification\FashionNet\Week2_dataset\clothes_dataset\dataset"
all_images, category, color = PreProcessing.load_data(path=path)    
trainX, testX, trainCategory, testCategory, trainColor, testColor = PreProcessing.split_data(all_images, category, color)

class FashionNet():
    def __init__(self, trainX, testX, trainCategory, testCategory, trainColor, testColor):
        self.trainX = trainX
        self.testX = testX
        self.trainCategory = trainCategory
        self.testCategory = testCategory
        self.trainColor = trainColor
        self.testColor = testColor


    def build(): # VGG16 architecture
        input_layer = layers.Input(shape=(128,128,3))