"""
Preprocessor for RIEGL image and mask data - Implementation with json_to_masks functionality.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, json
from PIL import ImageColor
import argparse
import pandas as pd
import glob
# Config will be passed from the orchestrator
import time
from typing import Tuple, Dict, Any, List
from app.utils.logger_utils import logger
import shutil
from tqdm import tqdm as tq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.deep_models.Algorithms.AMCNN.v1.config import AMCNN_V1_CONFIG
from app.deep_models.Algorithms.AMCNN.v1.utils import check_dir_exists

class maskGenerate():

    def __init__(self, combining_classes,classes_rgb):


        self.class_to_rgb = dict()

        for count in range(len(combining_classes)):
            for combo in combining_classes[count]:
                self.class_to_rgb[combo] = classes_rgb[count]
    

    def color_code_dict(self, csv_path):

        df=pd.read_excel(csv_path)
        return dict(zip(df.labelCode, df.colorCode))
        # color_dict

    def draw_poly(self, img, cordinates, colorCode):
        pts = np.array(cordinates, np.int32).reshape((-1, 1, 2))
        img=cv2.fillPoly(img, pts=[pts],  color=colorCode )

        return img

    def create_mask(self, data_2,imgshape):

        mask = np.ones(imgshape) * 255
        mask =  mask.astype('int32')
        for i in range(len(data_2)):
            # print(i)
            if data_2[i]['deleted']==True:
                colors = [255,255,255]
            else:
                if data_2[i]['label'] not in self.class_to_rgb.keys():

                    #print(data_2[i]['doughCode'],' not in ',self.class_to_rgb.keys())
                    continue

                colors = self.class_to_rgb[ data_2[i]['label'] ]
                # colors = ImageColor.getcolor('#'+str(colors), "RGB")

            curr=data_2[i]['coordinates']
            colorCode = [colors[2], colors[1], colors[0] ]

            thickness = int(data_2[i]['thickness'])
            if data_2[i]['type'] == 'Polygon':
                # print('drawing polygons')
                mask = self.draw_poly(mask, curr, colorCode)

            else:
                # print('drawing lines')
                for k in range(len(curr)-1):
                    mask = cv2.line(mask, pt1=tuple(curr[k]), pt2=tuple(curr[k+1]), color=colorCode,\
                                     thickness= thickness )

        return mask

    def create_masks(self, inputPath, outPath,imgs_path):

        jsonfiles = glob.glob(os.path.join(inputPath,'*.json'))
        logger.info(f'Processing {len(jsonfiles)} annotation files to generate masks')
        
        for file in tq(jsonfiles, desc="Generating masks"):

            try:

                if not file.endswith('.json'):
                    raise Exception(f'The encountered file is not json: {file}')

                file_name = os.path.split(file)[-1]

                imgpath = os.path.join(imgs_path,file_name.replace('_mask.json', '.jpg'))
                # print(imgpath)
                img = cv2.imread(imgpath)

                file_object = open(file)
                data = json.load(file_object)
                data_2 =  data['lsfs']['99']['objects']
                #print(file_path)
                mask = self.create_mask(data_2,img.shape)
                # print(np.unique(mask,return_counts=True))
                save_file_path = os.path.join(outPath, file_name.replace('json', 'png').replace('_mask', '') )
                cv2.imwrite(save_file_path, mask)

            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue


class RIEGLPreprocessor:
    """Preprocessor for RIEGL image and mask data with json_to_masks functionality."""
    
    def __init__(self):
        # logger.info("RIEGLPreprocessor.__init__() method called - preprocessor initialized")
        
        # Initialize global ConfigurationDict for masks_to_patches functionality
        global ConfigurationDict
        config_obj = AMCNN_V1_CONFIG()
        
        # Initialize mask generator with config
        # Use the values from the config and add missing label
        combining_classes = config_obj.combining_classes
        classes_rgb = config_obj.classes_rgb
        self.mask_generator = maskGenerate(combining_classes, classes_rgb)
        ConfigurationDict = {
            'image_size': config_obj.image_size,
            'base_patch_size': config_obj.base_patch_size,
            'tile_sizes': config_obj.tile_sizes,
            'same_size_images': config_obj.same_size_images,
            'mask_extension': config_obj.mask_extension,
            'datatype': config_obj.datatype
        }
        
        # Initialize color mappings for patches
        self.class_to_code = config_obj.class_to_code
        self.color_code = config_obj.color_code
    
    def preprocess_images_and_annotations(self, image_paths: List[str], annotation_paths: List[str], project_id: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Preprocess images and generate masks from annotations using json_to_masks and masks_to_patches functionality.
        
        Args:
            image_paths: List of image file paths
            annotation_paths: List of annotation file paths
            project_id: Project ID for organizing output files
            
        Returns:
            Tuple of (processed_images, generated_masks)
        """
        # logger.info(f"Starting preprocessing for {len(image_paths)} image-annotation pairs")
        
        # Extract paths from the first pair to determine directory structure
        if image_paths and annotation_paths:
            first_image_path = image_paths[0]
            first_json_path = annotation_paths[0]
            
            # Determine dataset path (parent directory containing jsons and orig_images)
            dataset_path = os.path.dirname(os.path.dirname(first_json_path))
            
            outputpath = os.path.join(dataset_path,'masks')
            json_folder_path = os.path.join(dataset_path,'jsons')
            orig_imgs_path = os.path.join(dataset_path,'orig_images')
            split_data_path = os.path.join(dataset_path,'split_patches_data')
            
            # logger.info(f"Dataset: {dataset_path}")
            
            # Step 1: Check directories and create masks
            check_dir_exists(json_folder_path, outputpath)
            
            # Start time measurement
            start_time = time.time()
            
            # Step 2: Generate masks using the mask generator
            logger.info("\nStep 1/2: Generating masks from annotations...")
            self.mask_generator.create_masks(json_folder_path, outputpath, orig_imgs_path)
            
            mask_generation_time = time.time() - start_time
            logger.info(f"✓ Masks generated in {mask_generation_time:.1f}s\n")
            
            # Step 3: Generate patches from masks (masks_to_patches functionality)
            logger.info("\nStep 2/2: Converting masks to training patches...")
            patch_start_time = time.time()
            
            # Setup split patches data directory
            if os.path.exists(split_data_path):
                shutil.rmtree(split_data_path)
            os.makedirs(split_data_path)
            
            # Create subdirectories for train/test/val splits
            data_subsets = ['train','test','val']
            classes_code = ['0','1','2','3','4']

            for subset_iter in data_subsets:
                for class_iter in classes_code:
                    path1 = os.path.join(split_data_path,subset_iter)
                    if not os.path.exists(path1):
                        os.makedirs(path1)
            
            images_list = glob.glob(os.path.join(orig_imgs_path,"*"))
            #masks_list = glob.glob(os.path.join(raw_masks_path,"*"))
            splitimgslist = [os.path.split(i)[1].split('.')[0] for i in images_list]
            masks_list = [os.path.join(outputpath,i+ConfigurationDict['mask_extension']) for i in splitimgslist]
            img_mask_mappings = list(zip(images_list, masks_list))
            train_cut = int(len(img_mask_mappings)*0.50)
            val_cut = int(len(img_mask_mappings)*0.25)
            train_mappings = img_mask_mappings[:train_cut]
            val_mappings = img_mask_mappings[train_cut:train_cut+val_cut]   
            test_mappings = img_mask_mappings[train_cut+val_cut:]
            # test_mappings = img_mask_mappings[train_cut+val_cut:]x
            
            logger.info(f"\nData split - Train: {len(train_mappings)}, Val: {len(val_mappings)}, Test: {len(test_mappings)}\n")
            
            # Generate patches for each split
            logger.info("Generating training patches...")
            target_dir = os.path.join(split_data_path,'train')
            self.Save_image_patches_for_training(train_mappings, target_dir, self.color_code)
            
            logger.info("\nGenerating validation patches...")
            target_dir = os.path.join(split_data_path,'val')
            self.Save_image_patches_for_training(val_mappings, target_dir, self.color_code)
            
            logger.info("\nGenerating test patches...")
            target_dir = os.path.join(split_data_path,'test')
            self.Save_image_patches_for_training(test_mappings, target_dir, self.color_code)
            
            patch_generation_time = time.time() - patch_start_time
            logger.info(f"\n✓ Patches generated in {patch_generation_time:.1f}s")
            
            # Calculate total time
            total_time = time.time() - start_time
            logger.info(f"\n✓ Preprocessing completed in {total_time:.1f}s\n")
            
        else:
            logger.warning("No image-annotation paths provided")
            return [], []
        
    def Save_image_patches_for_training(self, img_mappings, target_dir, color_code):
        base_patch_size = ConfigurationDict['base_patch_size']
        subset_name = os.path.basename(target_dir)
        
        # Process each image and save as bundle
        for img_idx, (img_path, mask_path) in enumerate(img_mappings):
            img_start_time = time.time()
            
            # Load image and mask
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            if img is None or mask is None:
                logger.warning(f"Skipping invalid image/mask pair: {img_path}")
                continue
            
            image_size = img.shape[:2]
            no_of_rows, no_of_cols = self.get_no_rows_cols(image_size, base_patch_size)
            num_patches = (no_of_rows - 1) * (no_of_cols - 1)
            
            # Collect all patches for this image
            padded_imgs_list, padding_tuples_list = self.get_padded_imgs(img, image_size)
            all_tiles = []  # List of tiles arrays (each is 5 multiscale tiles)
            all_labels = []
            all_indices = []
            
            # Process all patches for this image using ThreadPoolExecutor
            max_workers = min(16, max(4, os.cpu_count() * 2))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for col_index1 in range(no_of_cols - 1):
                    for row_index1 in range(no_of_rows - 1):
                        future = executor.submit(
                            self.process_tile_for_bundle,
                            row_index1, col_index1, padded_imgs_list, 
                            padding_tuples_list, mask, color_code, base_patch_size
                        )
                        futures.append((future, row_index1, col_index1))
                
                # Collect results
                for future, row_index1, col_index1 in futures:
                    try:
                        tiles_list, classcode = future.result()
                        if tiles_list is not None and classcode is not None:
                            all_tiles.append(tiles_list)
                            all_labels.append(int(classcode))
                            all_indices.append([row_index1, col_index1])
                    except Exception as e:
                        logger.error(f"Patch processing error at ({row_index1}, {col_index1}): {e}")
            
            # Save bundle for this image
            if all_tiles:
                # Get image stem for bundle filename
                img_stem = os.path.splitext(os.path.basename(img_path))[0]
                bundle_path = os.path.join(target_dir, f"{img_stem}.npz")
                
                # Convert to numpy arrays
                # tiles: (N, K, H, W, C) where N=num_patches, K=5 (tile sizes), H=W=base_patch_size, C=3
                tiles_array = np.array(all_tiles, dtype=np.uint8)  # Shape: (N, 5, 50, 50, 3)
                labels_array = np.array(all_labels, dtype=np.uint8)  # Shape: (N,)
                indices_array = np.array(all_indices, dtype=np.int32)  # Shape: (N, 2)
                
                # Save bundle with compression
                np.savez_compressed(
                    bundle_path,
                    tiles=tiles_array,
                    labels=labels_array,
                    indices=indices_array,
                    base_patch_size=base_patch_size,
                    tile_sizes=ConfigurationDict['tile_sizes']
                )
                
                img_total_time = time.time() - img_start_time
                logger.info(f"Image {img_idx+1}/{len(img_mappings)} ({img_stem}): {len(all_tiles)} patches saved in {img_total_time:.2f}s")
        
        # logger.info(f"\n✓ {subset_name} patch generation completed\n")  # commented per request

    def get_no_rows_cols(self,img_shape,base_patch_size):
        img_h_col,img_w_row = img_shape[0],img_shape[1]
        no_of_rows = int(img_w_row/base_patch_size)
        if int(img_w_row%base_patch_size)!=0:
            no_of_rows = int(img_w_row/base_patch_size)+1
        no_of_cols = int(img_h_col/base_patch_size)
        if int(img_h_col%base_patch_size)!=0:
            no_of_cols = int(img_h_col/base_patch_size)+1
        return no_of_rows,no_of_cols

    '''Input parameters tile size list and image. Return images with additional reuired padding '''
    def get_padded_imgs(self,img,image_size):

        tile_sizes = ConfigurationDict['tile_sizes']
        #image_size = ConfigurationDict['image_size']
        base_patch_size = ConfigurationDict['base_patch_size']
        padded_imgs_list = []
        padding_tuples_list = []
        #calculating padded images for each tile size
        for i in range(len(tile_sizes)):
            tilesize = tile_sizes[i]
            top, bottom, left, right = self.get_padding_dims(base_patch_size,tilesize,image_size)
            padding_tuples = (top, bottom, left, right)
            padding_tuples_list.append(padding_tuples)
            #print(top, bottom, left, right)
            padded_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,0)
            padded_imgs_list.append(padded_image)
        return padded_imgs_list,padding_tuples_list

    def get_padding_dims(self,base_patch_size, tile_size, image_size ) -> ('top_padding','bottom_padding','left_padding','right_padding'):

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

    '''Tiles of different sizes'''
    def get_tiles_of_all_sizes(self, row_index1, col_index1, padded_imgs_list, padding_tuples_list):
        tile_sizes = ConfigurationDict['tile_sizes']
        base_patch_size = ConfigurationDict['base_patch_size']
        tiles_list = []
        for i in range(len(tile_sizes)):
            tilesize = tile_sizes[i]
            tiles_or_tile = self.get_tile(
                row_index1, col_index1, base_patch_size, tilesize,
                padded_imgs_list[i], padding_tuples_list[i], padded_check=True
            )
            tile = tiles_or_tile
            if isinstance(tile, list):
                tile = tile[0] if len(tile) > 0 else None  # unwrap row dimension
            if isinstance(tile, list):
                tile = tile[0] if len(tile) > 0 else None  # unwrap size dimension
            if tile is None or not isinstance(tile, np.ndarray):
                logger.info(f"inference: bad tile from preprocessor i={i} type={type(tile)}")
                continue
            tile = cv2.resize(tile, (base_patch_size, base_patch_size))
            tiles_list.append(tile)
        return tiles_list

    def get_mask_tile_class(self, mask, color_code):
        # Convert color_code values to tuples if they are in string format
        for key in color_code:
            if isinstance(color_code[key], str):
                color_code[key] = eval(color_code[key])

        # Calculate unique colors and their counts
        colors, counts = np.unique(mask.reshape(-1, 3), axis=0, return_counts=True)
        colors = colors[:, ::-1]

        # If there's only 1 color, return its class directly
        if len(colors) == 1:

            most_frequent_color = tuple(colors[0])
            return next((key for key, value in color_code.items() if value == most_frequent_color), None)

        if len(colors) > 2:
            #print('more than 2 colors',colors)
            # Exclude class 0 and 1 colors for the decision
            excluded_indices = [i for i, color in enumerate(colors) if tuple(color) in [color_code['0'], color_code['1']]]
            colors = np.delete(colors, excluded_indices, axis=0)
            counts = np.delete(counts, excluded_indices)

        if len(colors) == 0:

            return '1'  # Return class 1 if only class 0 and 1 colors were present and now excluded

        # Find the index of the most frequent color after exclusions
        max_index = np.argmax(counts)
        most_frequent_color = tuple(colors[max_index])

        # Match the most frequent color with the color code dictionary and return its class
        for key, value in color_code.items():
            if value == most_frequent_color:
                return key

        # If no match found, return None
                return None
        

    def save_tiles(self,all_size_tiles,target_dir,classcode,counter):
        classpath = os.path.join(target_dir,str(classcode))
        os.makedirs(classpath,exist_ok=True)
        filename = os.path.join(classpath,str(counter)+'_'+ConfigurationDict['datatype']+'.npz')
        np.savez(filename, alltiles=all_size_tiles)

    # ... (rest of your code)
    # def process_tile(self, row_index1, col_index1, padded_imgs_list, padding_tuples_list, mask, color_code, base_patch_size, counter, target_dir):
    #     # Perform the processing for a single tile
    #     all_size_tiles = self.get_tiles_of_all_sizes(row_index1, col_index1, padded_imgs_list, padding_tuples_list)
    #     masktile_result = self.get_tile(row_index1, col_index1, base_patch_size, base_patch_size, mask, padding_tuple=None, padded_check=False)
        
    #     # Extract the actual numpy array from the list structure
    #     if isinstance(masktile_result, list) and len(masktile_result) > 0:
    #         masktile = masktile_result[0]  # Get first row
    #         if isinstance(masktile, list) and len(masktile) > 0:
    #             masktile = masktile[0]  # Get first tile size
    #     else:
    #         masktile = masktile_result
        
    #     classcode = self.get_mask_tile_class(masktile, color_code)
    #     self.save_tiles(all_size_tiles, target_dir, classcode, counter)

    def process_tile_for_bundle(self, row_index1, col_index1, padded_imgs_list, padding_tuples_list, mask, color_code, base_patch_size):
        """Process a single tile and return tiles + class code (for bundle format)."""
        try:
            # Get multiscale tiles
            all_size_tiles = self.get_tiles_of_all_sizes(row_index1, col_index1, padded_imgs_list, padding_tuples_list)
            
            # Get mask tile for class determination
            masktile_result = self.get_tile(row_index1, col_index1, base_patch_size, base_patch_size, mask, padding_tuple=None, padded_check=False)
            
            # Extract mask tile
            if isinstance(masktile_result, list) and len(masktile_result) > 0:
                masktile = masktile_result[0]
                if isinstance(masktile, list) and len(masktile) > 0:
                    masktile = masktile[0]
            else:
                masktile = masktile_result
            
            # Get class code
            classcode = self.get_mask_tile_class(masktile, color_code)
            
            # Return tiles list and class code
            return all_size_tiles, classcode
        except Exception as e:
            logger.error(f"Error processing tile ({row_index1}, {col_index1}): {e}")
            return None, None
        
    def get_tile(self,row_index, col_index, base_patch_size, tile_size, padded_img, padding_tuple, padded_check=True):
        # Ensure row_index is 1-D array even if scalar
        row_index_arr = np.atleast_1d(row_index)
        x = (row_index_arr * base_patch_size) + (base_patch_size / 2)
        y = (col_index * base_patch_size) + (base_patch_size / 2)

        if not isinstance(tile_size, list):
            tile_size = [tile_size]
        tile_size = np.array(tile_size)

        if padded_check:
            top, bottom, left, right = padding_tuple
            y = y + top
            x = x + left

        half_tile_size = tile_size / 2
        full_col_tiles = []
        for k in range(len(row_index_arr)):
            tiles_of_sizes = []
            for i in range(len(tile_size)):
                tile = padded_img[
                    int(y - half_tile_size[i]) : int(y + half_tile_size[i]),
                    int(x[k] - half_tile_size[i]) : int(x[k] + half_tile_size[i]),
                    :
                ]
                tile = cv2.resize(tile, (base_patch_size, base_patch_size))
                tiles_of_sizes.append(tile)
            full_col_tiles.append(tiles_of_sizes)

        return full_col_tiles