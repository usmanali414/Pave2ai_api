from keras.utils import to_categorical  # type: ignore
from keras.utils import Sequence  # type: ignore
import random, os, glob, cv2
import numpy as np
from app.deep_models.Algorithms.AMCNN.v1 import utils


class DataGen(Sequence):
     
    def __init__(self, image_size, class_folder_dict, dataset_dir,  batch_size=200, subset='train', shuffle_check=True):
        
        
        self.image_size = image_size
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.class_folder_names_dict = class_folder_dict
        self.batch_size = batch_size
        self.n = 0
        self.shuffle_check = shuffle_check
        #self.on_epoch_end()
        self.images_path = []
        self.labels = []
        self.read_imagesPath_list()
        if self.shuffle_check:
            self.shuffle_imageslist()
        self.max = self.__len__()
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.images_path):
            self.batch_size = len(self.images_path) - index*self.batch_size

        images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
        labels = self.labels[index*self.batch_size : (index+1)*self.batch_size]

        images_batch = []
        #labels_batch = []
        x0,x1,x2,x3,x4 = [],[],[],[],[]
        for i in range(len(images_path)):
            ## Read image and mask
            
            tiles1 = utils.parse_images(images_path[i])
            
            x0.append(cv2.cvtColor(tiles1[0], cv2.COLOR_BGR2RGB ))
            x1.append(cv2.cvtColor(tiles1[1], cv2.COLOR_BGR2RGB ))
            x2.append(cv2.cvtColor(tiles1[2], cv2.COLOR_BGR2RGB ))
            x3.append(cv2.cvtColor(tiles1[3], cv2.COLOR_BGR2RGB ))
            x4.append(cv2.cvtColor(tiles1[4], cv2.COLOR_BGR2RGB ))
            

        x0 = np.array(x0 ) /255.
        x1 = np.array(x1 ) /255.
        x2 = np.array(x2 ) /255.
        x3 = np.array(x3 ) /255.
        x4 = np.array(x4 ) /255.
        onehot_labels = to_categorical(labels,int(len(list(self.class_folder_names_dict.keys()))))
    
        return [x0,x1,x2,x3,x4], np.array(onehot_labels)
    
    def read_imagesPath_list(self):
        basepath = os.path.join(self.dataset_dir,self.subset)
        print(f"Looking for data in: {basepath}")
        
        total_files_found = 0
        for key1 in list(self.class_folder_names_dict.keys()):
            class_path = os.path.join(basepath,key1,'*.npz')
            print(f"Looking for files in: {class_path}")
            
            class_imgs_list = glob.glob(class_path)
            print(f"Found {len(class_imgs_list)} files in class {key1}")
            
            if self.subset == 'train' and str(key1) == '0':
                cutt_tokeep = len(class_imgs_list)/4
                class_imgs_list  = class_imgs_list[:int(cutt_tokeep)]
                print(f"Reduced to {len(class_imgs_list)} files for training class 0")
            
            self.images_path.extend(class_imgs_list)
            self.labels.extend([int(self.class_folder_names_dict[key1])]  * int(len(class_imgs_list)) )
            total_files_found += len(class_imgs_list)
        
        print(f"Total files found for {self.subset}: {total_files_found}")
        
    def on_epoch_end(self):
        # random.shuffle(self.images_path)
        # random.shuffle(self.masks_path)
        self.shuffle_imageslist()
        pass
    
    def shuffle_imageslist(self):
        c = list(zip(self.images_path, self.labels))
        random.shuffle(c)
        self.images_path,self.labels= zip(*c)
        
    def __len__(self):
        return int(np.ceil(len(self.images_path)/float(self.batch_size)))
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
