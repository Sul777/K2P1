import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import cropping_face as cf
import resizing_img_224 as res
from google.colab.patches import cv2_imshow

def swapping_face(changed_img_path, data_path_list, image, data_name, img_index, boxes):

  data_crop224_imgs_path = data_path_list[2]
  data_ada_imgs_path = data_path_list[4]
  data_ada224_imgs_path = data_path_list[5]
  data_swapped224_imgs_path = data_path_list[8]

  img2 = image

  for i in range(len(img_index)):
    # 1. 원본 이미지에서 얼굴 crop
    newboxes = cf.cropping_face(boxes[int(img_index[i])], img2, 0.025, data_name, 224, img_index[i], data_crop224_imgs_path)

    # 2. styleGAN2-ADA로 생성한 얼굴 이미지 224사이즈로 resizing
    stimg_path = data_ada_imgs_path + '/morphing_results_' + img_index[i] + '.png'
    res.resizing_img_224(stimg_path, data_ada224_imgs_path, img_index[i])

    # 3. SIM SWAP으로 얼굴 교체(a=styleGAN2 224 사이즈 얼굴 이미지, b=원본 224 사이즈 얼굴 이미지)
    pic_a_path = data_ada224_imgs_path + '/morphing_results224_' + img_index[i] + '.png'
    pic_b_path = data_crop224_imgs_path + '/croppedImage224_' + img_index[i] + '.png'
    cmd_simswap = f'python /content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/SimSwap/test_one_image.py --use_mask --pic_a_path {pic_a_path} --pic_b_path {pic_b_path}'
    os.system(cmd_simswap)

    # 4. 교체된 얼굴 이미지 화질 개선
    in_dir = data_swapped224_imgs_path +'/croppedImage224_' + img_index[i] + '_result.jpg'
    #fe.Enhancement(in_dir, data_swapped224_imgs_path, data_name)
    cmd_gpen = f'python /content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/GPEN/face_enhancement.py {in_dir} {data_swapped224_imgs_path} {data_name}'
    os.system(cmd_gpen)

    # 5. 화질이 개선된 얼굴 이미지를 불러와서 명암 & 채도 조절
    swapped_image = cv2.imread(data_swapped224_imgs_path + '/croppedImage224_' + img_index[i] + '_result_GPEN.jpg', cv2.IMREAD_COLOR)
    #cv2_imshow(swapped_image)

    # 명암 조절
    val = 8
    ar = np.full(swapped_image.shape, (val, val, val), dtype=np.uint8)
    sub = cv2.subtract(swapped_image, ar)
    #cv2_imshow(sub)

    # 채도 조절
    sub = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
    subimg = Image.fromarray(sub)
    change_color = ImageEnhance.Color(subimg)
    color_output = change_color.enhance(0.88)
    #display(color_output)

    # 6. newbox의 좌표값에 바뀐 이미지 박스를 붙임
    croppedimage = np.array(color_output)
    croppedimage = cv2.cvtColor(croppedimage, cv2.COLOR_BGR2RGB)
    croppedimage = cv2.resize(croppedimage, (newboxes[2]-newboxes[0], newboxes[3]-newboxes[1]), interpolation = cv2.INTER_CUBIC)

    # 7. 원본 이미지에 사각형으로 마스킹하여 swap된 얼굴 갖다 붙이기
    # 얼굴 마스크를 만들기 위해 img2_face_mask의 모양대로 img2_head_mask 생성
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_face_mask = np.zeros_like(img2_gray)
    pl = 10

    # 검은 배경에 하얀 얼굴 모양의 마스크 생성
    img2_head_mask = cv2.rectangle(img2_face_mask, (newboxes[0]+pl, newboxes[1]+pl), (newboxes[2]-pl, newboxes[3]-pl), (255, 255, 255), -1)
    #cv2_imshow(img2_head_mask)

    # 얼굴 모양의 마스크 생성(검은 얼굴)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    #cv2_imshow(img2_face_mask)

    # 원본 이미지에서 얼굴 모양 마스크를 사용하여 검은 얼굴로 뚫어놓기
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    #cv2_imshow(img2_head_noface)
    center_face2 = (int(((newboxes[0]+pl) + (newboxes[2]-pl)) / 2), int(((newboxes[1]+pl) + (newboxes[3]-pl)) / 2))

    # 명암, 채도 조절된 이미지의 테두리 부분을 일부 제외하고 crop
    height, width, channels = croppedimage.shape
    croppedimage = Image.fromarray(croppedimage)        # numpy 배열 형태인 croppedimage를 이미지 형태로 변환
    croppedimage = croppedimage.crop((pl, pl,  height-pl, width-pl))

    # img2 크기만한 까만 배경 생성
    img2_height, img2_width, img2_channels = img2.shape
    img2_new_face = np.zeros((img2_height, img2_width, img2_channels), np.uint8)

    # img2 까만 배경에 새로 생성한 얼굴 갖다 붙이기
    img2_new_face = Image.fromarray(img2_new_face)        # numpy 배열 형태인 img2를 이미지 형태로 변환
    img2_new_face.paste(croppedimage, (newboxes[0]+pl, newboxes[1]+pl))    # 새 얼굴을 좌표값에 갖다 붙임
    img2_new_face = np.array(img2_new_face)               # img2 이미지를 numpy 배열 형태로 다시 변환
    #cv2_imshow(img2_new_face)

    img2 = cv2.seamlessClone(img2_new_face, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    #cv2_imshow(img2)

    # 8. 마지막 사람인 경우 결과물 저장
    if (i == len(img_index)-1) :
      cv2.imwrite(changed_img_path + '/' + data_name +  '_result.png', img2)
      return (img2)

