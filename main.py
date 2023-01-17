from deep_net import FashionNet
from data_preprocessing import PreProcessing
import utility



path = "D:\Python_Codes\Deep_Learning\Advanced Tensorflow\Multi_Label Classification\FashionNet\Week2_dataset\clothes_dataset\dataset"
all_images, category, color = PreProcessing.load_data(path=path)    
trainX, testX, trainCategory, testCategory, trainColor, testColor = PreProcessing.split_data(all_images, category, color)