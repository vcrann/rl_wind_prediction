import numpy as np
import keras
import random


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        data,
        lookback_window,
        prediction_window,
        shuffle=True,
        batch_size=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data = data
        self.lookback_window = lookback_window
        self.prediction_window = prediction_window
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, batch_index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.list_IDs[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]

        X = np.empty((self.batch_size, self.lookback_window, 3))
        Y = np.empty((self.batch_size, self.prediction_window, 3))
        # print(indexes)
        # test1, test2 = self.__create_datapoint(indexes[0])
        # print(type(test1))
        # print(type(test2))
        i = 0
        for index in indexes:
            # X[index], Y[index] = self.__create_datapoint(index)
            x, y = self.__create_datapoint(index)
            # print(index)
            # print(y.shape)
            # X[i], Y[i] = self.__create_datapoint(index)
            X[i] = x
            Y[i] = y
            i += 1

        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        # self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            random.shuffle(self.list_IDs)
        #     np.random.shuffle(self.indexes)

    def __create_datapoint(self, index):
        lookback = self.data[index - self.lookback_window + 1 : index + 1][:]
        # current plus 199 previous values
        # lookback = [
        #     self.data[0][index - lookback_window + 1 : index + 1],
        #     self.data[1][index - lookback_window + 1 : index + 1],
        #     self.data[2][index - lookback_window + 1 : index + 1],
        # ]  # current plus 199 previous values

        target = self.data[index + 1 : index + self.prediction_window + 1][:]
        # 20 next values, gives (t+1, t+2, ..., t+prediction_window)
        # target = [
        #     self.data[0][index + 1 : index + prediction_window + 1],
        #     self.data[1][index + 1 : index + prediction_window + 1],
        #     self.data[2][index + 1 : index + prediction_window + 1],
        # ]  # 20 next values, gives (t+1, t+2, ..., t+prediction_window)

        return lookback, target
