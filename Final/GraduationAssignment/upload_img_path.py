from pathlib import Path

def upload_img_path(data_name):

  # 데이터별로 RetinaFace detected crop image, aligned image, stGAN2-ADA projection image, Face swapped image를 위한 폴더들 생성
  #collapse-hide
  # data_crop_img_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/RetinaFace-Cropimgs/' + data_name)     # 데이터별로 크롭된 얼굴 이미지를 저장할 경로
  # data_crop224_img_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/RetinaFace-Cropimgs/' + data_name + '/crop-224')     # 데이터별로 224 사이즈로 크롭된 얼굴 이미지를 저장할 경로
  # data_aligned_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/Aligned_imgs/' + data_name)        # 데이터별로 정렬된 얼굴 이미지를 저장할 경로
  # data_ada_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/ADA-Results/' + data_name)             # 데이터별로 stGAN2-ADA로 만든 얼굴 이미지를 저장할 경로
  # data_ada224_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/ADA-Results/' + data_name + '/crop-224')             # 데이터별로 stGAN2-ADA로 만든 얼굴 이미지를 저장할 경로
  # data_swapped224_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/Final_ing_mine/Swappedimgs/' + data_name)      # 데이터별로 swapped된 얼굴 이미지를 저장할 경로

  data_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/Processeddata/' + data_name)     # 데이터 관련 이미지들을 저장할 경로
  data_crop_imgs_path = data_path/'RetinaFace-Cropimgs'     # 데이터별로 크롭된 얼굴 이미지를 저장할 경로
  data_crop224_imgs_path = data_path/'RetinaFace-Cropimgs/crop-224'     # 데이터별로 224 사이즈로 크롭된 얼굴 이미지를 저장할 경로
  data_aligned_imgs_path = data_path/'Aligned_imgs'        # 데이터별로 정렬된 얼굴 이미지를 저장할 경로
  data_ada_imgs_path = data_path/'ADA-Results'              # 데이터별로 stGAN2-ADA로 만든 얼굴 이미지를 저장할 경로
  data_ada224_imgs_path = data_path/'ADA-Results/crop-224'                # 데이터별로 stGAN2-ADA로 만든 얼굴 이미지를 저장할 경로
  data_ada_generate_imgs_path = data_path/'ADA-Results/generate'    # generate.py을 실행하여 랜덤 이미지 생성을 위한 폴더 생성  
  data_ada_projection_imgs_path = data_path/'ADA-Results/projection'    # projection.py을 실행하여 생성된 이미지를 저장하기 위한 폴더 생성
  data_swapped224_imgs_path = data_path/'Swappedimgs'      # 데이터별로 swapped된 얼굴 이미지를 저장할 경로

  # Make Images folder if it doesn't exist.
  if not data_path.exists():
      data_path.mkdir()

  # Make Crop Images folder if it doesn't exist.
  if not data_crop_imgs_path.exists():
      data_crop_imgs_path.mkdir()

  # Make Crop 224 size Images folder if it doesn't exist.
  if not data_crop224_imgs_path.exists():
      data_crop224_imgs_path.mkdir()
      
  # Make Aligned Images folder if it doesn't exist.
  if not data_aligned_imgs_path.exists():
      data_aligned_imgs_path.mkdir()

  # Make ADA Images folder if it doesn't exist.
  if not data_ada_imgs_path.exists():
      data_ada_imgs_path.mkdir()

  # Make ADA Crop 224 size Images folder if it doesn't exist.
  if not data_ada224_imgs_path.exists():
      data_ada224_imgs_path.mkdir()
  
  # Make Generation Images folder if it doesn't exist.
  if not data_ada_generate_imgs_path.exists():
    data_ada_generate_imgs_path.mkdir()

  # Make Projection Images folder if it doesn't exist.
  if not data_ada_projection_imgs_path.exists():
    data_ada_projection_imgs_path.mkdir()

  # Make Swapped 224 size Images folder if it doesn't exist.
  if not data_swapped224_imgs_path.exists():
      data_swapped224_imgs_path.mkdir()

  data_path = str(data_path)
  data_crop_imgs_path = str(data_crop_imgs_path)
  data_crop224_imgs_path = str(data_crop224_imgs_path)
  data_aligned_imgs_path = str(data_aligned_imgs_path)
  data_ada_imgs_path = str(data_ada_imgs_path)
  data_ada224_imgs_path = str(data_ada224_imgs_path)
  data_ada_generate_imgs_path = str(data_ada_generate_imgs_path)
  data_ada_projection_imgs_path = str(data_ada_projection_imgs_path)
  data_swapped224_imgs_path = str(data_swapped224_imgs_path)
      
  print(data_path, '\n', data_crop_imgs_path, '\n', data_crop224_imgs_path, '\n', data_aligned_imgs_path)
  print(data_ada_imgs_path, '\n', data_ada224_imgs_path, '\n', data_ada_generate_imgs_path, '\n', data_ada_projection_imgs_path, '\n', data_swapped224_imgs_path)
  
  return data_path, data_crop_imgs_path, data_crop224_imgs_path, data_aligned_imgs_path, data_ada_imgs_path, data_ada224_imgs_path, data_ada_generate_imgs_path, data_ada_projection_imgs_path, data_swapped224_imgs_path