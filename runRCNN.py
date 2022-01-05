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
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("download**************")


class ShapesConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + (Horse and Man)
    # Number of training steps per epoch

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (  32, 64 , 128 ,256,512)  # anchor side in pixels

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

def list2array(list):
    b = np.array(list[0])
    for i in range(1, len(list)):
        b = np.append(b, list[i],axis=0)
    return b
 
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存txt文件成功")


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

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
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
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def test_model():
    # 基礎設定
    dataset_root_path = r"/home/user/TODGX/test_data"
    img_floder = os.path.join(dataset_root_path, "pic")
    mask_floder = os.path.join(dataset_root_path, "cv2_mask")
    # yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder) #照片張數
    # print(imglist)
    count = len(imglist)
    # print(count)

    # train與val資料集準備
    # dataset_train = DrugDataset()
    # dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    # dataset_train.prepare()
    # display_image_samples(dataset_train)

    # print("dataset_train-->",dataset_train._image_ids)

    dataset_test = DrugDataset()
    dataset_test.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_test.prepare()  

    inference_config = InferenceConfig()
    model_path = "/home/user/Mask_RCNN/mask_rcnn_coco_0060.h5"

    model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=MODEL_DIR)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    print(dataset_test.image_ids)
    img_list = np.random.choice(dataset_test.image_ids, 85)
    APs = []
    count1 = 0

    for image_id in img_list:
    # 加载测试集的ground truth
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, inference_config,image_id, use_mini_mask=False)
    # 将所有ground truth载入并保存
        if count1 == 0:
            save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
        else:
            save_box = np.concatenate((save_box, gt_bbox), axis=0)
            save_class = np.concatenate((save_class, gt_class_id), axis=0)
            save_mask = np.concatenate((save_mask, gt_mask), axis=2)
 
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
 
    # 启动检测
        results = model.detect([image], verbose=0)
        r = results[0]
 
    # 将所有检测结果保存
        if count1 == 0:
            save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
        else:
            save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
            save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
            save_score = np.concatenate((save_score, r["scores"]), axis=0)
            save_m = np.concatenate((save_m, r['masks']), axis=2)
 
        count1 += 1
 
# 计算AP, precision, recall
    AP, precisions, recalls, overlaps = \
            utils.compute_ap(save_box, save_class, save_mask,save_roi, save_id, save_score, save_m,iou_threshold=0.5)
 
    print("AP: ", AP)
    print("mAP: ", np.mean(AP))

    plt.plot(recalls, precisions, 'b', label='PR')
    plt.title('precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    text_save('Kpreci.txt', precisions)
    text_save('Krecall.txt', recalls)





# inference_config = InferenceConfig()
# model = modellib.MaskRCNN(mode="inference",
#                           config=inference_config,
#                           model_dir=MODEL_DIR)
# model_path = os.path.join(MODEL_DIR, "KL1000.h5")  # 修改成自己训练好的模型
 
# # Load trained weights
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)
 
# img_list = np.random.choice(dataset_test.image_ids, 85)
# APs = []


if __name__ == "__main__":
    test_model()
    # test_model, inference_config = load_test_model(class_number)
    # test_random_image(test_model, dataset_val, inference_config)
    
