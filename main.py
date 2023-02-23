from deep_net import FashionNet
from data_preprocessing import PreProcessing
from utility import plot_show

# load and preprocess the data
path = "D:\Python_Codes\Deep_Learning\Advanced Tensorflow\Multi_Label Classification\FashionNet\Week2_dataset\clothes_dataset\dataset"
all_images, category, color = PreProcessing.load_data(path=path)    
trainX, testX, trainCategory, testCategory, trainColor, testColor = PreProcessing.split_data(all_images, category, color)

# create and train the FashionNet model
model = FashionNet(trainX, testX, trainCategory, testCategory, trainColor, testColor, epochs=40)
trained_model, history = model.build()

# plot the training and validation accuracy
plot_show.plot_accuracy(history, "category_output")
plot_show.plot_accuracy(history, "color_output")

# plot the training and validation loss
plot_show.plot_loss(history, "category_output")
plot_show.plot_loss(history, "color_output")
