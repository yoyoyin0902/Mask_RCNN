import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib,utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image





ROOT_DIR = os.path.abspath("/home/user/Mask_RCNN/")
# ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


iter_num = 0

# Path to trained weights file
COCO_MODEL_PATH = "/home/user/Mask_RCNN/mask_rcnn_coco.h5"

class ShapesConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name    NAME = "shapes"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + (Horse and Man)
    # Number of training steps per epoch

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16,32, 64, 128,256)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE =100
    
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9

    VALIDATION_STEPS = 20

config = ShapesConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# def display_image_samples(dataset_train):
#     # Load and display random samples
#     image_ids = np.random.choice(dataset_train.image_ids, 4)

#     for image_id in image_ids:
#         image = dataset_train.load_image(image_id)
#         mask, class_ids = dataset_train.load_mask(image_id)
#         visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

class DrugDataset(utils.Dataset):
    # 得到該圖中有多少個例項（物體）
    def get_obj_index(self,image):
        n = np.max(image)
        return n
 
    # 解析labelme中得到的yaml檔案，從而得到mask每一層對應的例項標籤
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        # print(info)
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(),Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels
    

    def draw_mask(self, num_obj, mask, image,image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_shapes_train(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes 增加類別
        self.add_class("shapes", 1, "pen") 
         
        for i in range(count):
            # 獲取圖片寬和高
            # print(i)

            filestr = imglist[i].split(".")[0]
            # print(filestr)
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder +"/" + filestr + ".png"
            # print(mask_path)
            
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"
            # print(yaml_path)
            # print(dataset_root_path + "/labelme_json/" + filestr + "_json/info.png")
            cv_img = cv2.imread(dataset_root_path +"/labelme_json/" + filestr + "_json/img.png")
            # print(cv_img.shape[1])
            # print(cv_img.shape[0])
            # print(cv_img)
            # path=img_floder +"/"+  imglist[i]
            self.add_image(source="shapes", image_id=i, path=img_floder +"/"+  imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
    
    def load_shapes_val(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes 增加類別
        self.add_class("shapes", 1, "pen") 
         
        for i in range(count):
            # 獲取圖片寬和高
            # print(i)

            filestr = imglist[i].split(".")[0]
            # print(filestr)
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder +"/" + filestr + ".png"
            # print(mask_path)
            
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"
            # print(yaml_path)
            # print(dataset_root_path_val + "/labelme_json/" + filestr + "_json/info.png")
            cv_img = cv2.imread(dataset_root_path +"/labelme_json/" + filestr + "_json/img.png")
            # print(cv_img.shape[1])
            # print(cv_img.shape[0])
            # print(cv_img)
            # path=img_floder +"/"+  imglist[i]
            self.add_image(source="shapes", image_id=i, path=img_floder +"/"+  imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        # print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        # print(len(labels))
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("pen") != -1:
                labels_form.append("pen")
            # elif labels[i].find("triangle")!=-1:
            #     #print "column"
            #     labels_form.append("triangle")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        # print("class_id: ",class_ids)
        return mask, class_ids.astype(np.int32)


def train_model():
    # 基礎設定
    dataset_root_path = r"/home/user/TODGX/data"
    dataset_root_path_val = r"/home/user/TODGX/data_val"

    img_floder = os.path.join(dataset_root_path, "pic")
    img_floder_val = os.path.join(dataset_root_path_val, "pic")

    mask_floder = os.path.join(dataset_root_path, "cv2_mask")
    mask_floder_val = os.path.join(dataset_root_path_val, "cv2_mask")
    # yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder) #照片張數
    imglist_val = os.listdir(img_floder_val)
    # print(imglist)
    count = len(imglist)
    count_val = len(imglist_val)
    # print(count_val)

    # train與val資料集準備
    dataset_train = DrugDataset()
    dataset_train.load_shapes_train(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()
    # display_image_samples(dataset_train)

    # print("dataset_train-->",dataset_train._image_ids)

    dataset_val = DrugDataset()
    dataset_val.load_shapes_val(count_val, img_floder_val, mask_floder_val, imglist_val, dataset_root_path_val)
    dataset_val.prepare()  

    # image_ids = np.random.choice(dataset_train.image_ids, 10)
    # for image_id in image_ids:
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


    # Create models in training mode
    # config = ShapesConfig()
    # config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # # Which weights to start with?
    # # 第一次訓練時，這裡填coco，在產生訓練後的模型後，改成last
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last models you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,
    #             epochs=60,
    #             layers='heads')

    # # Fine tune all layers
    # # Passing layers="all" trains all layers. You can also
    # # pass a regular expression to select which layers to
    # # train by name pattern.
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=10,
    #             layers="all")
    
    
    
    # x_epoch, y_tra_loss, y_val_loss = modellib.call_back()
    # loss_visualize(x_epoch, y_tra_loss, y_val_loss)

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# def load_test_model(num_classes):
#     inference_config = InferenceConfig(num_classes)

#     # Recreate the model in inference mode
#     model = modellib.MaskRCNN(mode="inference",
#                               config=Config,
#                               model_dir=MODEL_DIR)

#     # Get path to saved weights
#     # Either set a specific path or find last trained weights
#     # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#     model_path = model.find_last()

#     # Load trained weights
#     print("Loading weights from ", model_path)
#     model.load_weights(model_path, by_name=True)
#     return model, inference_config

# def load_inference_model(num_classes, model_path):
#     inference_config = InferenceConfig(num_classes)

#     # Recreate the model in inference mode
#     model = modellib.MaskRCNN(mode="inference",
#                               config=inference_config,
#                               model_dir=model_path)

#     # Get path to saved weights
#     # Either set a specific path or find last trained weights
#     # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#     #model_path = model.find_last()

#     # Load trained weights
#     print("Loading weights from ", model_path)
#     model.load_weights(model_path, by_name=True)
#     return model, inference_config


def predict():
    import skimage.io
    from mrcnn import visualize
    model_path = "/home/user/Mask_RCNN/mask_rcnn_shapes_0050.h5" #/home/user/Mask_RCNN/mask_rcnn_shapes_0050.h5
    # model.load_weights(model_path, by_name=True)
    print(model_path)

    IMAGE_DIR = "/home/user/test/Mask_RCNN/images"
    print(IMAGE_DIR)
    
    # sys.path.append(os.path.join(ROOT_DIR,"sample/coco/"))
    # import coco

    # Create models in training mode
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    # model_path = model.find_last()
    

    # Load trained weights (fill in path to trained weights here)
    # assert model_path = "/home/user/test/Mask_RCNN/mask_rcnn_shapes_0030.h5"   # , "Provide path to trained weights"
    model.load_weights(model_path, by_name=True)
    # print("Loading weights from ", model_path)

    class_names = ['BG', 'pen'] 

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2] # next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    # image = skimage.io.imread(file_names)

    # a = datetime.now()
    print("----------")

    # Run detection
    results = model.detect([image], verbose=1)

    # b= datetime.now()
    # print("spend time: ",(b-a).seconds)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,
                                         r['scores'])

   

    #rois:檢測到對象的邊框
    #masks:檢測到對方的遮罩



if __name__ == "__main__":
    train_model()
    # test_model, inference_config = load_test_model(class_number)
    # test_random_image(test_model, dataset_val, inference_config)
    # predict()
  
