from tensorflow import keras
import matplotlib.pyplot as plt
import io
from PIL import Image

plt.rcParams['font.family'] = 'Arial'

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}


    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('sparse_categorical_accuracy'))


    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('sparse_categorical_accuracy'))


    def loss_plot(self, loss_type):

        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        print("Training accuracy:")
        print(self.accuracy[loss_type])
        acc = plt.plot(iters, self.accuracy[loss_type], 'r', label='train accuracy')
        print("Training loss:")
        print(self.losses[loss_type])
        # loss
        loss = plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True) # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('accuracy and loss')
        plt.title('Training accuracy-loss curves of the CNN model')
        plt.legend(loc="upper right")

        # png1=io.BytesIO()
        # plt.savefig(png1,format='png',dpi=500,pad_inches=.1,bbox_inches='tight')

        # png2=Image.open(png1)
        # png2.save('training.tiff')
        # png1.close()
        plt.show()
