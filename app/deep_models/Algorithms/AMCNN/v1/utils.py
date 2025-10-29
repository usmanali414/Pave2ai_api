import os, re, cv2
import numpy as np
import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def check_imgfile_validity(folder, file):
    """Function to check if the files in the given path are valid image files.
    
    Args:
        folder (str): path containing the image files
        filenames (list): a list of image filenames
    Returns:
        valid_files (bool): True if all the files are valid image files else False
        msg (str): Message that has to be displayed as error
    """
    
    full_file_path = os.path.join(folder, file)
    regex = "([^\\s]+(\\.(?i:(jpe?g|png)))$)"
    p = re.compile(regex)

    if not os.path.isfile(full_file_path):
        
        new_file = file.replace('png', 'jpg') if 'png' in file else file.replace('jpg', 'png')
        new_file_path = os.path.join(folder, new_file)
        if not os.path.isfile(new_file_path):
            return False, f"Mask for {file} not found: "
        return True, new_file_path
    if not (re.search(p, file)):
        return False, "Invalid image file: " + file
    return True, full_file_path


def check_dir_exists(folderPath, msg):
  if not os.path.exists(folderPath):
      raise Exception(f"Error:  {msg} does not exist.")

def parse_images(imgpath):
    data = np.load(imgpath)
    return data['alltiles']


def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight


def get_callbacks(csv_checkpoint_name, model_checkpoint_name):

    csv_logger = CSVLogger(csv_checkpoint_name, append=False)
    checkpoint = ModelCheckpoint(model_checkpoint_name, verbose=1, save_best_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9, verbose=1)

    return [csv_logger, checkpoint]

def check_dir_exists(inputPath, outPath):
    if not os.path.exists(inputPath):
        raise Exception("Error:  Input folder does not exist.")

    if not os.path.exists( outPath ):
        os.makedirs(outPath )