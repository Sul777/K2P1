from retinaface.pre_trained_models import get_model
from PIL import Image
import cv2
import numpy as np
import cropping_face as cf

def detecting_face(image, data_name, data_crop_img_path):
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()
    faces = model.predict_jsons(image)

    # 'bbox'좌표만 추출
    boxes = list()
    for i, face in enumerate(faces):
        #print(i, face['bbox'])
        #print(i, faces[face]['bbox'])
        #print(i, faces[face])
        boxes.append(face['bbox'])

    print('boxes : ', boxes)

    crop_file_num_temp = 0
    #crop_path = '/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/RetinaFace-Cropimgs/'
    boxes_list, newboxes, croppedimage_list = cf.cropping_face(boxes, image, 0.3, data_name, 1024, crop_file_num_temp, data_crop_img_path)

    return faces, boxes, boxes_list, newboxes, croppedimage_list
