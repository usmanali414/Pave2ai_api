import warnings
warnings.filterwarnings('ignore')
import os

import time


from app.deep_models.Algorithms.AMCNN.v1.config import AMCNN_V1_CONFIG
from app.deep_models.Algorithms.AMCNN.v1.model import modelMCNN
from app.deep_models.Algorithms.AMCNN.v1.data_generators import DataGen
from app.deep_models.Algorithms.AMCNN.v1 import utils
import tensorflow as tf
from tqdm import tqdm as tq
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def reading_data(split_data_path, class_folder_dict, batch_size):

    image_size = None   
    train_gen = DataGen(image_size, class_folder_dict, split_data_path,  batch_size, subset='train', shuffle_check=True)
    test_gen = DataGen(image_size,class_folder_dict, split_data_path, batch_size, subset='test', shuffle_check=False)
    val_gen = DataGen(image_size,class_folder_dict, split_data_path, batch_size, subset='val', shuffle_check=False)

    print('Total training samples: ',len(train_gen)*batch_size)
    print('Total validation samples: ',len(val_gen)*batch_size)
    print('Total testing samples: ',len(test_gen)*batch_size)

    return train_gen, val_gen, test_gen


if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="path to input image folder json masks are placed",type=str)
   
    args = parser.parse_args()


    configs = config.configurations()
    split_data_path = os.path.join(args.dataset_path,configs.train_data_path) #'../accumulated data/olddrmp_new_patch_data'

    train_gen, val_gen, test_gen = reading_data(split_data_path, configs.class_folder_dict,\
                                                configs.modelConfg['batch_size'])

    learningRate = configs.modelConfg['learning_rate']
    optim = Adam(learningRate)
    model_mcnn = modelMCNN(configs.labels_dict, configs.modelConfg, configs.base_patch_size)
    model = model_mcnn.classifier(optim, is_training=True)

    modelpath = os.path.join(os.getcwd(),configs.model_path)
    if not os.path.exists( modelpath ):
        print('created folder at', configs.model_path )
        print("=====================================")
        os.makedirs(modelpath )
                
    model_checkpoint_name = os.path.join(modelpath, configs.modelname+'.h5')
    csv_checkpoint_name = os.path.join(modelpath, configs.modelname+'.csv')



    def get_callbacks(csv_checkpoint_name, model_checkpoint_name):

        csv_logger = CSVLogger(csv_checkpoint_name, append=False)
        checkpoint = ModelCheckpoint(model_checkpoint_name, verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9, verbose=1)

        return [csv_logger, checkpoint]

    #early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    callbacks = get_callbacks(csv_checkpoint_name, model_checkpoint_name)#, early_stopping]
    model.fit_generator(train_gen,
            validation_data = val_gen,
            steps_per_epoch = len(train_gen),
            validation_steps = len(val_gen),
            epochs = configs.modelConfg['N_epoch'],
            callbacks = callbacks,
            class_weight= model_mcnn.class_weights_dict)
        

        
    # Calculate Total Time
    total_time = time.time() - start_time

    # Print the Results
    print(f"Script execution time: {total_time} seconds")