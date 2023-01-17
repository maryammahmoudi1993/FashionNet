import deep_net
import utility
from sklearn.model_selection import train_test_split
import cv2 as cv 
from glob import glob
import numpy as np

path = "D:\Python_Codes\Deep_Learning\Advanced Tensorflow\Multi_Label Classification\FashionNet\Week2_dataset\clothes_dataset\dataset"

def load_data(path): # load data and preprocessing (normalize, resize)
    all_images = []
    category = []
    color = []
    for i, im in enumerate(glob(path+ "\*\*")):
        img = cv.imread(im)
        img = cv.resize(img, (128, 128))
        all_images.append(img)
        category.append(im.split("\\")[-2].split('_')[1]) 
        color.append(im.split("\\")[-2].split('_')[0])

        if i % 100 == 0:
            print("{} percent of 2500 data has been loaded".format(i))

    all_images = np.array(all_images, dtype=float)/255.0
    return all_images, category, color
all_images, category, color = load_data(path=path)    

def split_data(all_images, category, color):
    split = train_test_split(all_images, category, color, test_size=0.2)
    (trainX, testX, trainCategory, testCategory, trainColor, testColor) = split
    return trainX, testX, trainCategory, testCategory, trainColor, testColor
trainX, testX, trainCategory, testCategory, trainColor, testColor = split_data(all_images, category, color)