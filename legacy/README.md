# Multi-Label Classification with FashionNet
# Maryam Mahmoudi
# In Progress
This repository contains Python code that demonstrates how to build a Multi-Label Classification model using FashionNet.

# Prerequisites
Python 3.x
Tensorflow 2.x
Keras
Scikit-learn
OpenCV
Dataset
The dataset used in this project is a collection of clothes images with corresponding labels for category and color. The dataset is organized as follows:

#dataset/
│
├── color_label_1_category_label_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── color_label_1_category_label_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── color_label_2_category_label_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── color_label_2_category_label_2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
#Usage
Clone the repository:

git clone https://github.com/<username>/Multi-Label-Classification-FashionNet.git
cd Multi-Label-Classification-FashionNet
# Load and preprocess data:

from data_preprocessing import PreProcessing
path = "path/to/dataset"
preprocess = PreProcessing(path)
all_images, category, color = preprocess.load_data(path)
category_labels, color_labels = preprocess.label_Bin(category, color)
trainX, testX, trainCategory, testCategory, trainColor, testColor = preprocess.split_data(all_images, category_labels, color_labels)
Build and train the model:


from model import FashionNet
fashion_net = FashionNet(trainX, testX, trainCategory, testCategory, trainColor, testColor, epochs=40)
model, history = fashion_net.build(train_data=[trainCategory, trainColor])
# Evaluate the model:

loss, cat_loss, col_loss, cat_acc, col_acc = model.evaluate(testX, [testCategory, testColor], verbose=0)
print("Total Loss: {:.2f}".format(loss))
print("Category Loss: {:.2f}".format(cat_loss))
print("Color Loss: {:.2f}".format(col_loss))
print("Category Accuracy: {:.2f}%".format(cat_acc * 100))
print("Color Accuracy: {:.2f}%".format(col_acc * 100))
# Acknowledgments
The dataset used in this project is a modified version of DeepFashion2 dataset.
