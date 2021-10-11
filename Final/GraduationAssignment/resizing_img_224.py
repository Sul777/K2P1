import cv2
from PIL import Image

def resizing_img_224(image_path, crop_path, img_index):
  #stGAN2-ADA로 만든 이미지 224사이즈로 resizing
  image = cv2.imread(image_path)

  img_resize = cv2.resize(image, (224,224), interpolation = cv2.INTER_CUBIC)
  img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
  #INTER_LINEAR는 빠르고 좋은편, INTER_CUBIC은 느리지만 품질이 가장 좋음
  croppedimage = Image.fromarray(img_resize)

  """
  crop_file_name = image_path.split('croppedImage')   # crop이미지 projection인 경우
  # image_path = /content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/Processeddata/data5/ADA-Results/aligned_croppedImage0_proj.png'
  #crop_file_name = image_path.split('seed')    # seed이미지 projection인 경우
  crop_file_num = crop_file_name[1].split('_proj')[0]
  # crop_file_name[1] = '0_proj.png'
  # crop_file_num = 0
  """
  #img_index = str(img_index)
  croppedimage.save(crop_path + '/morphing_results224_' + img_index + '.png', quality=95)     # crop이미지 projection인 경우
  #croppedimage.save(crop_path + '/croppedImage224_' + crop_file_num + '.png', quality=95)     # crop이미지 projection인 경우
  #croppedimage.save(crop_path + '/croppedImage224_' + crop_file_num + '_' + img_index +'.png', quality=95)   # seed이미지 projection인 경우
  # 데이터 별 폴더에 크롭된 이미지 저장

  #cv2_imshow(image)