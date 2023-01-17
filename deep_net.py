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
        x = layers.Conv2D(64, (3,3), activation="relu")(input_layer)
        x = layers.Conv2D(64, (3,3), activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(128, (3,3), activation="relu")(x)
        x = layers.Conv2D(64, (3,3), activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(256, (3,3), activation="relu")(x)
        x = layers.Conv2D(256, (3,3), activation="relu")(x)
        x = layers.Conv2D(256, (3,3), activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(512, (3,3), activation="relu")(x)
        x = layers.Conv2D(512, (3,3), activation="relu")(x)
        x = layers.Conv2D(512, (3,3), activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(512, (3,3), strides=(2,2), activation="relu")(x)
        x = layers.Conv2D(512, (3,3), strides=(2,2), activation="relu")(x)
        x = layers.Conv2D(512, (3,3), strides=(2,2), activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Flatten()(x)
        cat_net = layers.Dense(4, activation="softmax")
        col_net = layers.Dense(3, activation="softmax")

        net = models.Model(inputs=input_layer, outputs=[cat_net, col_net], name="FashionNet")

        losses = {
            "category_output":"categorical_crossentropy",
            "color_output":"categorical_crossentropy"
        }

        loss_weight = {"category_output":1.0, "color_output":1.0}

        net.compile(optimizer="adam", loss=losses, loss_weights=loss_weight, metrics=["accuracy"])
        return net

    def load_net(net):
        H = net.fit(x = trainX, y = {"category_output": trainCategory, "color_output": trainColor},
                    validation_data=(testX,{"category_output":testCategory, "color_output":testColor}), 
                    epochs = 40, verbose = 1)
        return H
        

