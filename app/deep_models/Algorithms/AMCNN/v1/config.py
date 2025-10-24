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
    modelname = 'experiment1_accumulated_checkpointV1'
    dataName = 'new_drmp'
    datatype='new'
    visible_device = "1"