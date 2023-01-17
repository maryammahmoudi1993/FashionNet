import deep_net
import utility
from sklearn.model_selection import train_test_split
import cv2 as cv 
from glob import glob
import numpy as np

path = "D:\Python_Codes\Deep_Learning\Advanced Tensorflow\Multi_Label Classification\FashionNet\Week2_dataset\clothes_dataset\dataset"

def load_data(path): # load data and preprocessing (normalize, resize)
    data = []
    for i, im in enumerate(glob(path+ "\*\*")):
        img = cv.imread(im)
        data.append(img)
    mx = np.max(data)
        

