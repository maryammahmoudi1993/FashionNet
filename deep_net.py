import tensorflow as tf  
from keras import layers, models
from data_preprocessing import PreProcessing



class FashionNet():
    epochs = 40
    def __init__(self, trainX, testX, trainCategory, testCategory, trainColor, testColor, epochs):
        self.trainX = trainX
        self.testX = testX
        self.trainCategory = trainCategory
        self.testCategory = testCategory
        self.trainColor = trainColor
        self.testColor = testColor
        self.epochs = epochs


    def build(self): # VGG16 architecture
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
        H = net.fit(x = self.trainX, y = {"category_output": self.trainCategory, "color_output": self.trainColor},
                    validation_data=(self.testX,{"category_output":self.testCategory, "color_output":self.testColor}), 
                    epochs=self.epochs , verbose = 1)
        net.save("FashionNet.h5")
        return net, H

    '''def load_net(self,net):
        H = self.net.fit(x = self.trainX, y = {"category_output": self.trainCategory, "color_output": self.trainColor},
                    validation_data=(self.testX,{"category_output":self.testCategory, "color_output":self.testColor}), 
                    epochs=self.epochs , verbose = 1)
        return H'''

    '''def save_model(self, net):
        net.save
        '''

