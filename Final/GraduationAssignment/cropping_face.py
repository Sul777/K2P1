from PIL import Image
import cv2
import numpy as np

def cropping_face(boxes, img2, plusnum, data_name, crop_size, crop_file_num, crop_path):
    rect_img2 = img2.copy()
    # cv2 gray numpy array로 구성된 img2를 이미지로 변환
    dimg = Image.fromarray(img2)

    boxes_list = [] # detection된대로 크롭한 이미지 리스트
    newboxes = []   # 정사각형 좌표값 리스트
    croppedimage_list = []  # 정사각형 좌표대로 크롭한 이미지 리스트

    for i, box in enumerate(boxes):
        if (crop_size == 224):
          box = boxes
        
        deimage = dimg.crop((box[0], box[1], box[2], box[3]))  # 이미지로부터 detection된 얼굴부분만 크롭
        boxes_list.append(np.array(deimage))  # 크롭된 얼굴 리스트에 저장

        xd = box[2] - box[0]
        yd = box[3] - box[1]
        d = 0

        if (xd >= yd):
            # x좌표 차이가 더 큰 경우 y좌표값을 x좌표 차이값 만큼 맞춰서 정사각형 생성
            d = int(xd / 2)
            plus = int(d * plusnum)
            d += plus
            ymean = int((box[1] + box[3]) / 2)
            #print(ymean)

            newboxes.append([box[0]-plus, ymean-d, box[2]+plus, ymean+d])   # 크롭할 얼굴의 새 좌표값 저장
            croppedimage = dimg.crop((box[0]-plus, ymean-d, box[2]+plus, ymean+d))  # 이미지로부터 해당 좌표에 맞는 얼굴부분을 정사각형으로 만들어서 크롭
            cv2.rectangle(rect_img2, (box[0]-plus, ymean-d), (box[2]+plus, ymean+d), (0,0,255), 15)

        else:
            # y좌표 차이가 더 큰 경우 x좌표값을 y좌표 차이값 만큼 맞춰서 정사각형 생성
            d = int(yd / 2)
            plus = int(d * plusnum)
            d += plus
            xmean = int((box[0] + box[2]) / 2)
            #print(xmean)

            newboxes.append([xmean-d, box[1]-plus, xmean+d, box[3]+plus])   # 크롭할 얼굴의 새 좌표값 저장     
            croppedimage = dimg.crop((xmean-d, box[1]-plus, xmean+d, box[3]+plus))  # 이미지로부터 해당 좌표에 맞는 얼굴부분을 정사각형으로 만들어서 크롭
            cv2.rectangle(rect_img2, (xmean-d, box[1]-plus), (xmean+d, box[3]+plus), (0,0,255), 15)

        img_array = np.array(croppedimage)
        croppedimage_list.append(img_array)  # 크롭된 얼굴 리스트에 저장

        # 크롭 이미지를 저장할 경우
        if (crop_size > 0):
          img_resize = cv2.resize(img_array, (crop_size, crop_size), interpolation = cv2.INTER_CUBIC)
          img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
          #INTER_LINEAR는 빠르고 좋은편, INTER_CUBIC은 느리지만 품질이 가장 좋음
          croppedimage = Image.fromarray(img_resize)

        #****
        #crop_file_name = image_path.split('croppedImage')
        #****
        if (crop_size == 224):
            # 데이터 별 폴더에 크롭된 이미지 저장
            croppedimage.save(crop_path + '/croppedImage224_' + crop_file_num + '.png', quality=95)
            return newboxes[0]
        elif (crop_size == 1024):
            #croppedimage.save('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/' + data_name + '/RetinaFace-Cropimgs/croppedImage' + str(i) + '.png', quality=95)
            croppedimage.save(crop_path + '/croppedImage' + str(i) + '.png', quality=95)
            if (len(boxes)-1):
              cv2.imwrite(crop_path + '/rectangle_origin_image.png', rect_img2)
    
    return boxes_list, newboxes, croppedimage_list
        