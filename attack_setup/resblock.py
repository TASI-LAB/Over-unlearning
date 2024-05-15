import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Activation

CIFAR_SHAPE = (32, 32, 3)

class ResnetBlock(Model):
    def __init__(self, channels: int, strides=1,down_sample=False):
        super().__init__()
        self.__channels = channels
        self.__down_sample = down_sample
        
        KERNEL_SIZE = (3, 3)
        INIT_SCHEME = "he_normal"
        self.conv_1 = Conv2D(self.__channels, strides=strides, kernel_size=KERNEL_SIZE, use_bias=False,
                             padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels,
                             kernel_size=KERNEL_SIZE, padding="same",use_bias=False, kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()
        if self.__down_sample:
            self.res_conv = Conv2D(self.__channels, strides=strides, use_bias=False,kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out

class ResNet(Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, kernel_size=3, padding='same', kernel_initializer="he_normal", use_bias=False)
        self.init_bn = BatchNormalization()
        # self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64, strides = 1)
        self.res_1_2 = ResnetBlock(64, strides = 1)
        self.res_2_1 = ResnetBlock(128, down_sample=True, strides=2)
        # self.res_2_2 = ResnetBlock(128, strides=1)
        # self.res_3_1 = ResnetBlock(256, down_sample=True, strides=2)
        # self.res_3_2 = ResnetBlock(256, strides=1)
        # self.res_4_1 = ResnetBlock(512, down_sample=True, strides=2)
        # self.res_4_2 = ResnetBlock(512, strides=1)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc1 = Dense(128, activation="softmax")
        self.fc_bn = BatchNormalization()
        self.fc2 = Dense(num_classes, activation="softmax")
    
    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        # out = self.pool_2(out)
        # for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
        #     out = res_block(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc_bn(out)
        out = self.fc2(out)
        return out

class ResBlock:
    def __init__(self, restore=None, session=None, use_log=False, num_classes=10):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = num_classes
        model = ResNet(self.num_labels)
        model.build(input_shape = (None,32,32,3))
        self.model = model
        if restore is not None:
            self.model.load_weights(restore)
        
    def predict(self, data):
        return self.model(data)

