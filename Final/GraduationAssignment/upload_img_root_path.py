from pathlib import Path

def upload_img_root_path():
  # RetinaFace 및 Face swap 후 저장될 이미지를 위한 폴더 생성(한번만 실행하여 상위 폴더 생성)

  #collapse-hide
  # 아래 3개 폴더는 나중에 데이터 별로 하위 폴더 생성
  # crop_img_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/RetinaFace-Cropimgs')     # 크롭된 얼굴 이미지를 저장할 경로
  # aligned_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/Aligned_imgs')        # 정렬된 얼굴 이미지를 저장할 경로
  # ada_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/ADA-Results')             # stGAN2-ADA로 만든 얼굴 이미지를 저장할 경로
  # swapped_imgs_path = Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/Swappedimgs')          # swapped된 얼굴 이미지를 저장할 경로

  # 결과물 폴더는 하나만 생성
  prepath = Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment')
  precessed_data_path = prepath/'Processeddata'
  changed_img_path = prepath/'Changedimgs'          # 바뀐 얼굴 이미지를 저장할 경로
  concat_img_path = prepath/'Concatimgs'            # 원본과 바뀐 얼굴 이미지를 같이 붙여서 저장할 경로


  # # Make Images folder if it doesn't exist.
  # if not crop_img_path.exists():
  #     crop_img_path.mkdir()

  # # Make Aligned Images folder if it doesn't exist.
  # if not aligned_imgs_path.exists():
  #     aligned_imgs_path.mkdir()

  # # Make ADA Images folder if it doesn't exist.
  # if not ada_imgs_path.exists():
  #     ada_imgs_path.mkdir()

  # # Make Swapped Images folder if it doesn't exist.
  # if not swapped_imgs_path.exists():
  #     swapped_imgs_path.mkdir()

  # Make Processed Images folder if it doesn't exist.
  if not precessed_data_path.exists():
      precessed_data_path.mkdir()

  # Make Changed Images folder if it doesn't exist.
  if not changed_img_path.exists():
      changed_img_path.mkdir()

  # Make Concat Images folder if it doesn't exist.
  if not concat_img_path.exists():
      concat_img_path.mkdir()

  changed_img_path = str(changed_img_path)
  concat_img_path = str(concat_img_path)

  #print(crop_img_path, '\n', aligned_imgs_path, '\n', ada_imgs_path, '\n', swapped_imgs_path, '\n', changed_img_path, '\n', concat_img_path)
  print(precessed_data_path, '\n', changed_img_path, '\n', concat_img_path)

  #return crop_img_path, aligned_imgs_path, ada_imgs_path, swapped_imgs_path, changed_img_path, concat_img_path
  return precessed_data_path, changed_img_path, concat_img_path