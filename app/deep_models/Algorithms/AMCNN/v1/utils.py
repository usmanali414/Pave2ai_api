import os, re, cv2
import numpy as np
import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger #type: ignore

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

# def parse_images(imgpath):
#     data = np.load(imgpath)
#     return data['alltiles']

def parse_images(imgpath, patch_idx=None, bundle_cache=None):
    """Parse images from bundle format or legacy individual .npz format.
    
    Args:
        imgpath: Path to bundle file or legacy .npz file
        patch_idx: Optional patch index for bundle format. If None, selects random patch.
        bundle_cache: Optional dict to cache loaded bundles (key=bundle_path, value={'tiles': ..., 'labels': ...})
    
    Returns:
        tuple: (tiles_array, label) where:
            - tiles_array: numpy array of shape (5, H, W, C) with 5 multiscale tiles
            - label: integer class label
    """
    # Check cache first for bundle format
    if bundle_cache is not None and imgpath in bundle_cache:
        cached_data = bundle_cache[imgpath]
        tiles = cached_data['tiles']
        labels = cached_data['labels']
    else:
        # Load from disk
        data = np.load(imgpath, allow_pickle=True)
        
        # Check if it's a bundle (has 'tiles' key) or legacy format (has 'alltiles' key)
        if 'tiles' in data:
            tiles = data['tiles']
            labels = data['labels']
            # Cache it if cache dict provided
            if bundle_cache is not None:
                bundle_cache[imgpath] = {
                    'tiles': tiles,
                    'labels': labels
                }
        else:
            # Legacy format: individual .npz file
            alltiles = data['alltiles']
            path_parts = imgpath.split(os.sep)
            label = 0
            for part in path_parts:
                if part in ['0', '1', '2', '3', '4']:
                    label = int(part)
                    break
            return alltiles, label
    
    # Handle bundle format
    n_patches = len(tiles)
    if n_patches == 0:
        raise ValueError(f"Bundle {imgpath} has no patches")
    
    # Use provided patch_idx or select random
    if patch_idx is None:
        patch_idx = np.random.randint(0, n_patches)
    elif patch_idx >= n_patches:
        patch_idx = patch_idx % n_patches  # Wrap around if out of bounds
    
    selected_tiles = tiles[patch_idx]  # Shape: (K, H, W, C) = (5, 50, 50, 3)
    selected_label = int(labels[patch_idx])
    
    return selected_tiles, selected_label


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