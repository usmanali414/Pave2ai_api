class AMCNN_V1_CONFIG():
    
    # color code and class mapping for new labels 
    #PLease write the values in BGR Format
    color_code = {'0':(255,255,255),'1':(54, 244, 67),'2':(143, 206, 0), '3':(201, 0, 118), '4':(244, 67, 54)}
    class_to_code = {'0':'Background','1':'Slabs','2':'Cracks','3':'Mark','4':'Patch'}

    #color code and class mapping for old drmp labelled images
    #class_to_code = {'0':'Background','1':'Joint','2':'Patch','3':'Crack'}
    #color_code  = {'0':  (255,  255,  255), '1':  (44,  160,  44), '2':  (14,  127,   255), '3':  (180,  119,  31)}

    train_data_path = "split_patches_data"
    
    
    combining_classes = [['conc-crk', 'conc-crk-seal', 'conc-cut', 'conc-spl'],['conc-mrk'],['conc-pat'],['conc-slb', 'bridge']]#,['conc-msk-long-ltrt']]
    classes_rgb = [(143, 206, 0), (201, 0, 118), (244, 67, 54),(54, 244, 67)]
    
    base_patch_size = 50
    tile_sizes = [50,350,500,1000,2500]
    same_size_images = False
    training_cut = 0.65
    validation_cut = 0.15
    image_size = (1308,2473)
    mask_extension = '.png'
    labels_dict = {0: 678131, 1: 3746977, 2: 107784, 3: 143241,4:5236}
    modelConfg = {'class_num':5,
              'learning_rate': 0.0001,#0.0001,
              'N_epoch':1,
              'batch_size':200}

    class_folder_dict = {'0':0,'1':1,'2':2,'3':3,'4':4}
    epochs=1
    model_path = "model_Logs"
    modelname = 'amcnn_weights'
    dataName = 'new_drmp'
    datatype='new'
    visible_device = "1"
    dataset_config = {
        "base_path": "static/AMCNN/v1",
        "images_dir": "orig_images",
        "jsons_dir": "jsons",
        "masks_dir": "masks",
        "patches_dir": "split_patches_data",
        "inference_base_dir": "inference",
        "inference_masks_dir": "masks",
        "inference_overlays_dir": "overlays"
    }
    
    # Dynamic static root (set by orchestrator); all AMCNN paths derive from this
    _STATIC_ROOT = None  # e.g., Path('E:/Pave2ai_api/static')

    @classmethod
    def set_static_root(cls, root_path):
        from pathlib import Path
        cls._STATIC_ROOT = Path(root_path)

    @classmethod
    def _default_static_root(cls):
        from pathlib import Path
        here = Path(__file__).resolve()
        # app/deep_models/Algorithms/AMCNN/v1/config.py → repo_root is parents[5]
        repo_root = here.parents[5] if len(here.parents) >= 6 else here.parents[-1]
        return repo_root / "static"

    @classmethod
    def get_static_root(cls):
        return cls._STATIC_ROOT or cls._default_static_root()

    @classmethod
    def get_model_root(cls):
        from pathlib import Path
        return cls.get_static_root() / Path("AMCNN") / "v1"

    @classmethod
    def get_dataset_root(cls):
        return cls.get_model_root() / "dataset"

    @classmethod
    def get_images_dir(cls):
        return cls.get_dataset_root() / cls.dataset_config["images_dir"]

    @classmethod
    def get_jsons_dir(cls):
        return cls.get_dataset_root() / cls.dataset_config["jsons_dir"]

    @classmethod
    def get_masks_dir(cls):
        return cls.get_dataset_root() / cls.dataset_config["masks_dir"]

    @classmethod
    def get_patches_dir(cls):
        return cls.get_model_root() / cls.dataset_config["patches_dir"]

    @classmethod
    def get_inference_base_dir(cls):
        return cls.get_model_root() / cls.dataset_config["inference_base_dir"]

    @classmethod
    def get_inference_masks_dir(cls):
        return cls.get_inference_base_dir() / cls.dataset_config["inference_masks_dir"]

    @classmethod
    def get_inference_overlays_dir(cls):
        return cls.get_inference_base_dir() / cls.dataset_config["inference_overlays_dir"]