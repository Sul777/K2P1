import os
import torch
import dnnlib
import legacy
import random
import numpy as np
import PIL
from pathlib import Path
from projector import run_projection

# 랜덤으로 생성된 이미지, crop 및 align 된 이미지 두개의 latent를 폴더내에서 뽑아내는 함수
def get_final_latents(data_ada_projection_imgs_path):
    #all_results = list(Path('/content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/Processeddata/data11/ADA-Results/projection').iterdir())
    all_results = list(Path(data_ada_projection_imgs_path).iterdir())
    aligned_final_latents = []
    seeds_final_latents = []

    latent_files = [x for x in all_results if 'projected_w' in x.name]
    aligned_latent_files = [x for x in latent_files if 'aligned' in x.name]
    seeds_latent_files = [x for x in latent_files if 'seed' in x.name]

    for index,value in enumerate(aligned_latent_files, start=0):
      print(index,value)
    for index,value in enumerate(seeds_latent_files, start=0):
      print(index,value)

    for file in aligned_latent_files:
        with open(file, mode='rb') as latent_pickle:
            aligned_final_latents.append(np.load(latent_pickle)['w'])

    for file_seed in seeds_latent_files:
        with open(file_seed, mode='rb') as latent_pickle_seed:
            seeds_final_latents.append(np.load(latent_pickle_seed)['w'])
 
    return aligned_final_latents, seeds_final_latents


def gen_proj_face(data_path_list, img_index, step_num):

  data_aligned_imgs_path = data_path_list[3]
  data_ada_imgs_path =  data_path_list[4]
  data_ada_generate_imgs_path = data_path_list[6]
  data_ada_projection_imgs_path = data_path_list[7]

  # 1000~9999의 값들 중 바꿀 얼굴 개수만큼만 랜덤 시드를 생성하여 styleGAN2-ADA의 projection에서 source로 시작할 수 있도록 만듦
  seeds = random.sample(range(1000, 10000), len(img_index))
  str_seeds = ''
  print(seeds)

  # generate.py의 입력을 위해 seed값 string으로 변환
  for i in range(len(seeds)):
    if(i == (len(seeds)-1)):
      str_seeds += str(seeds[i])
    else:
      str_seeds += str(seeds[i]) + ','

  print(str_seeds)


  # generate.py 실행을 통해 이미지 생성
  NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
  cmd_generate = f"python /content/drive/MyDrive/ColabNotebooks/Final/GraduationAssignment/stylegan2-ada-pytorch/generate.py --outdir={data_ada_generate_imgs_path} --trunc=1 --seeds={str_seeds} --network={NETWORK}"
  #!{cmd_generate}
  os.system(cmd_generate)

  #학습 횟수
  num_steps = step_num

  # random image 와 각각의 aligned image를 latent space로 투영시킵니다.
  # 즉, 각각의 이미지에대한 latent vector를 획득합니다. (.npz file)
  for i in range(len(img_index)):
    print('Projecting image %d/%d ...' % (i, len(img_index)))
    target_fname = str(data_aligned_imgs_path) + '/aligned_croppedImage'+ str(img_index[i]) +'.png'
    target_seeds = str(data_ada_generate_imgs_path) + '/seed' + str(seeds[i]) + '_' + str(i) + '.png'
    # run_projection(타겟 이미지, 결과물이 저장될 경로, 시드값, 학습 횟수)
    run_projection(target_fname=target_fname, outdir=data_ada_projection_imgs_path, num_steps=num_steps)
    run_projection(target_fname=target_seeds, outdir=data_ada_projection_imgs_path, num_steps=num_steps)

  # 모델 불러오기
  device = torch.device('cuda')
  with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print("SUCCCESS IN LOAD!")

  # 투영된 이미지의 latent vector들 불러오기
  aligned_final_latents, seeds_final_latents = get_final_latents(data_ada_projection_imgs_path)

  # latent vector 간 보간하기 (torch version으로 변경)
  interpol_vector = []
  ws = []
  for i in range(len(img_index)):
    interpol = aligned_final_latents[i]*(0.3) + seeds_final_latents[i]*(0.7)
    interpol_vector.append(interpol)
    w_s = torch.tensor(interpol, device=device)
    ws.append(w_s)

  outdir = data_ada_imgs_path
  noise_mode = 'const'
  for i,temp_w_s in enumerate(ws):
    for idx, w in enumerate(temp_w_s):
      img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
      img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
      print(i)
      img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/morphing_results_{str(img_index[i])}.png')


  print("ALL TASK IS FINISHED!")