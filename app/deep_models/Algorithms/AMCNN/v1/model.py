import tensorflow as tf
import os
import numpy as np
# import tensorflow_addons as tfa  # Commented out for inference
from tensorflow.keras.layers import GlobalAveragePooling2D,Reshape,Dense,Multiply,Conv2D,BatchNormalization
from tensorflow.keras.layers import Activation,Add,MaxPooling2D,UpSampling2D,Concatenate,Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from app.deep_models.Algorithms.AMCNN.v1 import utils

# #multigpu
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus[1:]:
#     tf.config.experimental.set_memory_growth(gpu, True)
# ###########

#multigpu
#strategy = tf.distribute.MirroredStrategy()
#########



class modelMCNN():

    def __init__(self, labels_dict, modelConfig, base_patch_size):

        self.class_weights_dict = utils.create_class_weight(labels_dict)
        self.modelConfg = modelConfig
        self.basePatchSize = base_patch_size
        self.model = None


    def weighted_categorical_crossentropy(self, y_true, y_pred):
        ## weights = [0.9,0.05,0.04,0.01]
        #def wcce(y_true, y_pred):
        Kweights = K.constant(list(self.class_weights_dict.values()))
        #if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)


    def conv2d(self, x, Kw, Kh, out_shape, name):
        c= tf.keras.layers.Conv2D(out_shape, (Kw,Kh), strides= (1,1), padding= "same", name= "Conv"+name)(x)
        c= tf.keras.layers.BatchNormalization(name= "BN"+name)(c)
        c= tf.keras.layers.ReLU(name="ReLU"+name)(c)
        return c
    def fully_connected(self,x, out_shape, name,kernel1=None):
        c = tf.keras.layers.Dense(out_shape, activation= "relu", name= "FC"+name,kernel_regularizer=kernel1)(x)
        return c

    def fully_connected_no_activation(self, x, out_shape, name):
        c = tf.keras.layers.Dense(out_shape, name= name)(x)
        return c

    def fully_connected_with_softmax(self, x, out_shape, name):
        c = tf.keras.layers.Dense(out_shape, name= "FC"+name)(x)
        c = tf.keras.layers.Softmax(name= "softmax_"+name)(c)
        return c

    #####################################Architecture################################
    def conv_net1(self, x):
        conv1 = self.conv2d(x, 3, 3, 32, "1_CNN1")
        conv2 = self.conv2d(conv1, 3, 3, 32, "2_CNN1")
        conv3 = self.conv2d(conv2, 3, 3, 64, "3_CNN1")
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name= "Maxpool1_CNN1")(conv3)
        return max_pool1

    def conv_net2(self, x):
        conv1 = self.conv2d(x, 3, 3, 32, "1_CNN2")
        conv2 = self.conv2d(conv1, 3, 3, 32, "2_CNN2")
        conv3 = self.conv2d(conv2, 3, 3, 64, "3_CNN2")
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name="Maxpool1_CNN2")(conv3)
        return max_pool1

    def conv_net3(self, x):
        conv1 = self.conv2d(x, 3, 3, 32, "1_CNN3")
        conv2 = self.conv2d(conv1, 3, 3, 32, "2_CNN3")
        conv3 = self.conv2d(conv2, 3, 3, 64, "3_CNN3")
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name="Maxpool1_CNN3")(conv3)
        return max_pool1

    def conv_net4(self, x):
        conv1 = self.conv2d(x, 3, 3, 32, "1_CNN4")
        conv2 = self.conv2d(conv1, 3, 3, 32, "2_CNN4")
        conv3 = self.conv2d(conv2, 3, 3, 64, "3_CNN4")
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name="Maxpool1_CNN4")(conv3)
        return max_pool1

    def conv_net5(self, x):
        conv1 = self.conv2d(x, 3, 3, 32, "1_CNN5")
        conv2 = self.conv2d(conv1, 3, 3, 32, "2_CNN5")
        conv3 = self.conv2d(conv2, 3, 3, 64, "3_CNN5")
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name="Maxpool1_CNN5")(conv3)
        return max_pool1

    def attention_module(self, features, name):
        conv1 = self.conv2d(features, 1, 1, 64, name= "1_Att"+name)
        conv2 = self.conv2d(conv1, 1, 1, 64, name= "2_Att"+name)
        conv3 = self.conv2d(conv2, 1, 1, 64, name= "3_Att"+name)
        scores= tf.keras.layers.Activation('sigmoid', name= "Maxpool_Att"+name)(conv3)
        return scores

    def classifier(self, optimizer1,is_training=True):

        #multigpu
        #with strategy.scope():
        # self.basePatchSize = ConfigurationDict['base_patch_size']
        x1 = tf.keras.Input(shape=(self.basePatchSize, self.basePatchSize,3 ), name='1-In')
        x2 = tf.keras.Input(shape=(self.basePatchSize, self.basePatchSize,3 ), name='2-In')
        x3 = tf.keras.Input(shape=(self.basePatchSize, self.basePatchSize,3 ), name='3-In')
        x4 = tf.keras.Input(shape=(self.basePatchSize, self.basePatchSize,3 ), name='4-In')
        x5 = tf.keras.Input(shape=(self.basePatchSize, self.basePatchSize,3 ), name='5-In')
        
        features1 = self.conv_net1(x1)
        features2 = self.conv_net2(x2)
        features3 = self.conv_net3(x3)
        features4 = self.conv_net4(x4)
        features5 = self.conv_net5(x5)

        scores1 = self.attention_module(features1, "_Scale1")
        scores2 = self.attention_module(features2, "_Scale2")
        scores3 = self.attention_module(features3, "_Scale3")
        scores4 = self.attention_module(features4, "_Scale4")
        scores5 = self.attention_module(features5, "_Scale5")

        weighted_features1 = tf.keras.layers.multiply([features1, scores1], name= "WF1")
        weighted_features2 = tf.keras.layers.multiply([features2, scores2], name= "WF2")
        weighted_features3 = tf.keras.layers.multiply([features3, scores3], name= "WF3")
        weighted_features4 = tf.keras.layers.multiply([features4, scores4], name= "WF4")
        weighted_features5 = tf.keras.layers.multiply([features5, scores5], name= "WF5")
        features_c = tf.keras.layers.concatenate([weighted_features1, weighted_features2, weighted_features3, weighted_features4, weighted_features5], name= "Conc_WF")
        
        conv1 = self.conv2d(features_c, 3, 3, 256, name= "1_DCNN")
        conv2 = self.conv2d(conv1, 3, 3, 256, name= "2_DCNN")
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name= "Maxpool1_DCNN")(conv2)

        conv4 = self.conv2d(max_pool1, 3, 3, 256, name= "3_DCNN")
        conv5 = self.conv2d(conv4, 3, 3, 512, name= "4_DCNN")
        max_pool2 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name= "Maxpool2_DCNN")(conv5)

        conv7 = self.conv2d(max_pool2, 3, 3, 512, name= "5_DCNN")
        conv8 = self.conv2d(conv7, 3, 3, 1024, name= "6_DCNN")
        max_pool3 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name= "Maxpool3_DCNN")(conv8)
        flatten_features = tf.keras.layers.Flatten(name= "flatten_DCNN")(max_pool3)

        fully_connected1 = self.fully_connected(flatten_features, 2048,  name= "1_DCNN")

        fully_connected2 = self.fully_connected(fully_connected1, 1024, name= "2_DCNN")
        fully_connected2 = tf.keras.layers.Dropout(0.2, name= "Drout1_DCNN")(fully_connected2, training=is_training)

        fully_connected3 = self.fully_connected(fully_connected2, 512, name= "3_DCNN")
        fully_connected3 = tf.keras.layers.Dropout(0.2, name= "Drout2_DCNN")(fully_connected3, training=is_training)

        fully_connected4 = self.fully_connected(fully_connected3, 256, name= "4_DCNN")
        fully_connected5 = self.fully_connected(fully_connected4, 128, name= "5_DCNN")
        outputs = self.fully_connected_with_softmax(fully_connected4, self.modelConfg['class_num'], \
                                                                            name= "FinalOutput")
        #return features1, features2, features3, features4, features5, scores1, scores2, scores3, scores4, scores5, weighted_features1, weighted_features2, weighted_features3, weighted_features4, weighted_features5, out
        
        
        model = Model([x1,x2,x3,x4,x5], outputs)
        model.compile(optimizer1,
                    loss=self.weighted_categorical_crossentropy, #tf.keras.losses.CategoricalCrossentropy(),
                    #loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])  # Removed F1Score to avoid tensorflow_addons dependency
        return model
