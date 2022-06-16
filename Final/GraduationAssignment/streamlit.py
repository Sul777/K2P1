import os
from importlib import reload          # 모듈 재import
#from retinaface.pre_trained_models import get_model
from matplotlib import pyplot as plt
import PIL.Image
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import dlib
import time
import random
from google.colab.patches import cv2_imshow
from google.colab import files

#collapse-hide
# styleGAN2-ADA에서 사용되는 라이브러리들
import sys
# import해서 사용할 .py파일들을 포함하고 있는 경로 추가
# import하여 쓰려고 하는 것의 상위 폴더 경로 작성해줘야 돌아감
sys.path.append('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment')
sys.path.append('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/stylegan2-ada-pytorch')
sys.path.append('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/SimSwap')
sys.path.append('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/GPEN')

import torch
import dnnlib
import legacy
import pickle
import ipywidgets as widgets
from pathlib import Path
from tqdm import tqdm
from align_face import align_face
from projector import run_projection

# 코드 .py화한 후 import
import upload_img_root_path as uirp
import upload_img_path as uip
import detecting_face as df
import cropping_face as cf
import resizing_img_224 as res
import swapping_face as sw
import gen_proj_face as gpf

import enum
from re import sub
import streamlit as st
from PIL import Image

#collapse-hide
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

crop_success = False;
step_success = False;

# box color
st.markdown("""
<style>
.css-1vgnld3 {
    font-size: 14px;
    color: rgb(255 255 255);
    margin-bottom: 7px;
    height: 1.5rem;
    vertical-align: middle;
    display: flex;
    flex-direction: row;
    -webkit-box-align: center;
    align-items: center;
}
</style>
""",unsafe_allow_html=True)

# name color
st.markdown("""
<style>
.css-1d391kg h1 {
  font-size: 1.5rem;
  font-weight: 600;
  color: white;
}
</style>
""",unsafe_allow_html=True)

# member color
st.markdown("""
<style>
.css-1d391kg h2 {
  font-size: 1.25rem;
  font-weight: 600;
  color: white;
}
</style>
""",unsafe_allow_html=True)



# -sidebar image 위치 조정 구문
st.markdown("""
<style>
.css-1d391kg {
    background-color: rgb(43, 64, 91);
    background-attachment: fixed;
    flex-shrink: 0;
    height: 100vh;
    overflow: auto;
    padding: 5rem 1rem;
    position: relative;
    transition: margin-left 300ms ease 0s, box-shadow 300ms ease 0s;
    width: 21rem;
    z-index: 100;
    margin-left: 0px;
}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<style>
.css-1e5imcs {
    display: flex;
    flex-direction: column;
    position: relative;
    margin: 0px 0px 2rem;
    margin: px 0px 1rem;
}
</style>
""",unsafe_allow_html=True)

# sidebar ------------------------------------------------------------------
side_col1, side_col2, side_col3, side_col4, side_col5 = st.sidebar.columns(5)

side_col2.image("/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/pnu.png", width=172 )
#st.sidebar.image("/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/pnu.png", width=150 )

step_course = st.sidebar.selectbox("Go step by step", ('INTRO', 'STEP1','STEP2','STEP3','STEP4','STEP5','STEP6', 'STEP7'))
st.sidebar.markdown("""<br><br>""", unsafe_allow_html=True)
st.sidebar.markdown("""# Team Name : K2P1""", unsafe_allow_html=True)
st.sidebar.markdown("""## Members : """, unsafe_allow_html=True)

st.sidebar.markdown("""<span style="color:white">**_Name : Kim Gyeong Ho_**<br>
**eamil : happymoney0726@pusan.ac.kr** </span> """, unsafe_allow_html=True)

st.sidebar.markdown("""<span style="color:white"> **_Name : Park Su Min_**<br>
**eamil : happymin7@pusan.ac.kr** </span>""", unsafe_allow_html=True)

st.sidebar.markdown("""<span style="color:white"> **_Name : Kim Sun Gyu_**<br>
**eamil : rlatjsrb10@pusan.ac.kr** </span>""", unsafe_allow_html=True)
# sidebar ------------------------------------------------------------------


#crop_img_path, aligned_imgs_path, ada_imgs_path, swapped_imgs_path, changed_img_path, concat_img_path = uirp.upload_img_root_path()
precessed_data_path, changed_img_path, concat_img_path = uirp.upload_img_root_path()

if step_course == 'INTRO':
  st.write("# Preserve your image naturally")
  st.markdown(""" - Don't use Mosaic or Stick anymore. We will change your precious photos naturally.""", unsafe_allow_html=True)
  st.image('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/camera.jpg',use_column_width=True)
  st.markdown(""" <br><br><br>""", unsafe_allow_html=True)

# ---------------------------------------------------STEP1--------------------------------------------
elif step_course == 'STEP1':
  st.title('1. Face Detection')
  st.markdown(""" - This is the stage to detect faces in input photos. We use the "Retina face" model to detect faces, upload a photo and you can see the results.""", unsafe_allow_html=True)

  file = st.file_uploader("Upload your Image!",type=['png', 'jpeg', 'jpg'])

  gan_flag = False
  fn_src = None
  if file is None:
      st.text("Please upload an image file")
  else:
      with open(os.path.join("/content/drive/MyDrive/ColabNotebooks/Final/data",file.name), "wb") as f:
        f.write(file.getbuffer())
      fn_src = str(file.name)
      imgpath = '/content/drive/MyDrive/ColabNotebooks/Final/data/'
      image = cv2.imread(imgpath + fn_src)
      data_name = fn_src.split(sep='.')
      data_name = data_name[0]
      crop_success = True

      img_for_path = imgpath + fn_src
      with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/img_for_path.pickle', 'wb') as f:
        pickle.dump(img_for_path, f)
        

      data_path, data_crop_imgs_path, data_crop224_imgs_path, data_aligned_imgs_path, data_ada_imgs_path, data_ada224_imgs_path, data_ada_generate_imgs_path, data_ada_projection_imgs_path, data_swapped224_imgs_path = uip.upload_img_path(data_name)
      data_path_list = [data_path, data_crop_imgs_path, data_crop224_imgs_path, data_aligned_imgs_path, data_ada_imgs_path, data_ada224_imgs_path, data_ada_generate_imgs_path, data_ada_projection_imgs_path, data_swapped224_imgs_path]
      with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/path.pickle', 'wb') as lf:
        pickle.dump(data_path_list, lf)

      faces, boxes, boxes_list, newboxes, croppedimage_list = df.detecting_face(image, data_name, data_crop_imgs_path)

      with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/img_for_boxes.pickle', 'wb') as f:
        pickle.dump(boxes, f)
        

      show_img = Image.open(file)
      col1, col2 = st.columns(2)
      col1.header("Original")
      col1.image(show_img, use_column_width = True)

      # After Face detection
      col2.header("Face Detection!")
      example = Image.open(data_crop_imgs_path+'/rectangle_origin_image.png')
      #example = show_img.convert('LA')
      col2.image(example, use_column_width = True)
      st.success("File upload and Face Detection successful.")
      gan_flag = True
      
# ---------------------------------------------------STEP2--------------------------------------------
elif step_course == 'STEP2':
  st.title('2. Crop the Face')
  st.markdown(""" - Detected faces are cropped from the original image and saved separately. Here's what the process looks like.""")
  st.image('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/cropface.JPG', use_column_width = True)
  if crop_success:
    st.success("Image cropping was successful.")

# ---------------------------------------------------STEP3--------------------------------------------
elif step_course == 'STEP3':
  st.title('3. Select Face')
  st.markdown(""" - This is the step to select the face you don't want to change. If selected, that face will not have any changes in the photo.""")
  with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/path.pickle', 'rb') as lf:
    path_list = pickle.load(lf)
  
  data_crop_imgs_path = path_list[1]
  DIR = data_crop_imgs_path
  filenames = os.listdir(DIR)
  img_paths = []
  img_arr = []

  # 삭제를 위한 요소들
  flag_remove = []

  # 이미지 경로를 하나의 리스트에 모두 담기
  for file in filenames:
      img_paths.append(str(DIR+"/"+file))

  img_paths.remove(data_crop_imgs_path + '/crop-224')
  img_paths.remove(data_crop_imgs_path + '/rectangle_origin_image.png')

  ID_COUNT = 0
  # 폴더에 있는 모든 이미지들을 4개씩 한 행으로하여 출력
  for i, path_img in enumerate(img_paths):
      show_img = Image.open(path_img)
      show_img = show_img.resize((128,128))
      img_arr.append(show_img)

      if((i+1)%4 == 0 or (i+1) == len(img_paths)) :
          img_col = st.columns(4)
          for index, img in enumerate(img_arr):
              img_id = "Select Face" + str(ID_COUNT) + " 🥕"
              img_col[index].image(img_arr[index], use_column_width = True)
              flag_remove.append(img_col[index].checkbox(img_id))
              
              ID_COUNT += 1

          img_arr.clear()


  c1, c2, c3, c4 = st.columns([3,1,1,1])

  submitted = c4.button("SUBMIT")

  submit_flag = False

  if submitted :
    for i, flag in enumerate(flag_remove):
      if flag :
        os.remove(img_paths[i])
        
      st.success("Selected images were excluded!")
      submit_flag =True

# ---------------------------------------------------STEP4--------------------------------------------
elif step_course == 'STEP4':
  st.title('4. Select Training Step')
  step_input = st.number_input("Enter a step number", min_value=5, max_value=120,step = 1)
  s1, s2, s3, s4 = st.columns([3,1,1,1])
  step_button = s4.button("CHECK")

  if step_button :
      st.markdown(f""" * Step number : {step_input}""" )
      f = open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/step_input.txt', 'w')
      print(step_input, file=f)
      f.close()
      st.success("The number of learning step has been entered.")
      step_success = True

# image resize and align

# ---------------------------------------------------STEP5--------------------------------------------
elif step_course == 'STEP5':
  st.title('5. Resize image and Face alignment')
  st.markdown(""" - Resize the existing cropped image to a certain size to create a new virtual face and <br>align the face in the resized images to face the front.""", unsafe_allow_html=True)
  st.markdown(""" - With the above preprocessing process, we can create a higher quality virtual face.""", unsafe_allow_html=True)
  st.image('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/ra2.jpg', use_column_width = True)

  if step_success :
    st.success("Image resize and Face alignment succeeded.")

# ---------------------------------------------------STEP6--------------------------------------------
elif step_course == 'STEP6':
  st.title('6. Face Generation')
  st.markdown(""" - Virtual face is created by inputting the preprocessed image to the stylegan2 ada model.""", unsafe_allow_html=True)
  st.image('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/vface.jpg', use_column_width = True)


# ---------------------------------------------------STEP7--------------------------------------------
else:
  st.title('7. Face Swap')
  st.markdown(""" - Generated virtual face and the face in the existing image are replaced..""", unsafe_allow_html=True)
  
  # 학습 횟수 불러오기
  f = open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/step_input.txt','r')
  step_input = f.read()
  step_input = int(step_input)
  f.close()

  # 경로 불러오기
  with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/path.pickle', 'rb') as lf:
    data_path_list = pickle.load(lf)

  data_path = data_path_list[0]
  data_crop_imgs_path = data_path_list[1]
  data_crop224_imgs_path = data_path_list[2]
  data_aligned_imgs_path = data_path_list[3]
  data_ada_imgs_path = data_path_list[4]
  data_ada224_imgs_path = data_path_list[5]
  data_ada_generate_imgs_path = data_path_list[6]
  data_ada_projection_imgs_path = data_path_list[7]
  data_swapped224_imgs_path = data_path_list[8]

  face_swap_flag = False
  # Align all of our images using a landmark detection model!
  # RetinaFace-Cropimgs 폴더로부터 png형식을 가진 파일 모두 가져오기
  all_imgs = list(Path(data_crop_imgs_path).glob('*.png'))
  all_imgs.remove(Path(data_crop_imgs_path + '/rectangle_origin_image.png'))

  img_index = []
  for img in all_imgs:
      align_face(str(img)).save(data_aligned_imgs_path + '/aligned_' + img.name)
      img_index.append(str(img).split('croppedImage')[1].split('.png')[0])    # 폴더에 존재하는 crop이미지들의 인덱스만 추출

  #gen_proj_face(데이터 경로 리스트, 얼굴 인덱스)
  gpf.gen_proj_face(data_path_list, img_index, step_input)
  st.success("Face Generation succeeded.")
  face_swap_flag = True
  
  with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/img_for_path.pickle', 'rb') as rf:
    img_for_path = pickle.load(rf)

  with open('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/streamlit/img_for_boxes.pickle', 'rb') as kf:
    boxes = pickle.load(kf)

  image = cv2.imread(img_for_path)

  basename = os.path.basename(img_for_path)
  data_name = os.path.splitext(basename)[0]
  
  resultimg = sw.swapping_face(changed_img_path, data_path_list, image, data_name, img_index, boxes)

  origin = Image.open(img_for_path)
  rimg = Image.open(changed_img_path + '/' + data_name + '_result.png')
  get_concat_h(origin, rimg).save(concat_img_path + '/' + data_name + '_origin_result_concat.png')

  final_path = changed_img_path + '/' + data_name + '_result.png'
  if face_swap_flag:
    st.image(final_path, use_column_width = True)
    st.success("Face Swap succeeded.")
