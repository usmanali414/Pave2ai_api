import os, re, cv2
import numpy as np
import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def get_padding_dims( base_patch_size, tile_size, image_size ) -> ('top_padding','bottom_padding','left_padding','right_padding'):

    '''
    base_patch_size:    'Int' basic patch size(/ by 10) that we are going to extract to input in model
    tile_size:          'Int' Tile size to calculate padding to extract tile of any specific size.
    image_size:         'Tuple' (Height, Width) Image height, Image width
    '''
    
    if base_patch_size%10 != 0 :
        raise Exception("Patch size is not divisible by 10..!!!")
    if tile_size%2 != 0 :
        raise Exception("Tile size is not divisible by 10..!!!")
    
    # if tile_size == base_patch_size:
    #   return (0,0,0,0)
    
    left_padding =  int((tile_size/2)-(base_patch_size/2))
    top_padding  =  int((tile_size/2)-(base_patch_size/2))


    H,W = image_size[0],image_size[1]
    
    if int(H%base_patch_size) == 0:
        bottom_padding = top_padding
    else:
        bottom_padding = int((tile_size/2) - ((H%base_patch_size)/2))
    if int(W%base_patch_size) == 0:
        right_padding =  left_padding
    else:
        right_padding = int((tile_size/2) - ((W%base_patch_size)/2))
    return top_padding,bottom_padding,left_padding,right_padding



def get_tile(row_index, col_index, base_patch_size, tile_size, padded_img, padding_tuple, padded_check=True):
    
    # print(row_index)
    row_index = np.array(row_index)
    x = (row_index*base_patch_size)+(base_patch_size/2)
    y = (col_index*base_patch_size)+(base_patch_size/2)
    
    if not type(tile_size) is list:
      tile_size = [tile_size]
    tile_size = np.array(tile_size)
    # print(half_tile_size)

    if padded_check == True:
        top,bottom,left,right = padding_tuple
        y = y+top
        x = x+left

    half_tile_size = tile_size/2
    # print(x)
    # exit()
    # print('row_len', len(row_index), x[0] )
    full_col_tiles = []
    for k in range(len(row_index)):
        tiles_of_sizes= []

        for i in range(len(tile_size)):

            tile = padded_img[int(y-half_tile_size[i]):int(y+half_tile_size[i]),\
                 int(x[k] - half_tile_size[i]):int(x[k] + half_tile_size[i]), :]
            # print('resize')
            tile = cv2.resize(tile,(base_patch_size, base_patch_size))
            # print('resized', len(full_col_tiles))
            tiles_of_sizes.append(tile)
        
        full_col_tiles.append(tiles_of_sizes)
    
    return full_col_tiles

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