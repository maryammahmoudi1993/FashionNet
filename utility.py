import matplotlib.pyplot as plt 
from deep_net import FashionNet

class plot_show():
    '''fashnet = FashionNet()
    H, net = fashnet.build()'''

    def __init__(self, H):
        self.H = H

    def plt_show(self,H):
        plt.plot(self.H.history["category_output_accuracy"], label="category acc")
        plt.plot(H.history["val_category_output_accuracy"], label="val category acc")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        plt.close()