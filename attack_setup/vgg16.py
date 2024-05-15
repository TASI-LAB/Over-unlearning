from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

class VGG16:
    def __init__(self, restore=None, session=None, use_log=False, num_labels=10):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = num_labels

        n_filters = [128, 128, 128, 128, 128, 128]
        conv_params = dict(activation='relu', kernel_size=3,
                        kernel_initializer='he_uniform', padding='same')

        model = Sequential()
        # VGG block 1
        model.add(Conv2D(filters=n_filters[0], input_shape=(32,32,3), **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[1], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        # VGG block 2
        model.add(Conv2D(filters=n_filters[2], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[3], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        # VGG block 3
        model.add(Conv2D(filters=n_filters[4], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[5], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # dense and final layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        model.add(Dense(units=self.num_labels, activation='softmax'))
            
        if restore:
            print('model path: ', restore)
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
    

        
    
