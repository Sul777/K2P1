# 2021 Pusan National University CSE Graduation Project

- team name : K2P1
- project name : Portrait rights protection system


# How to use?

Bring this project to the Google Colab. Set the hardware accelerator to GPU in the runtime type of Google Colab. (Select **Runtime > Change runtime type > Hardware accelerator > GPU**)\
Open the file named '2021-PNU-K2P1-Portrait rights protection system.ipynb' on the Google Colab. 

0. Bring and install the necessary libraries.
```python
!pip install requests
!pip install Pillow
!pip install tqdm
!pip install dlib
!pip install -U retinaface_pytorch
!pip install ninja
!pip install insightface==0.2.1 onnxruntime moviepy
!pip install streamlit
!pip install pyngrok
!pip install "opencv-python-headless<4.3"
```

1. Restart the Google Colab runtime and then run the following code. (Select **Runtime > Restart runtime**) And allow drive mount permission.
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Sign up [ngrok](https://ngrok.com/) and get your authtoken. Then write and execute your authtoken after '!ngrok'.
```python
!ngrok authtoken write_your_authtoken_here
```

3. Execute the below codes. And click the link of 'p_url'.
```python
from pyngrok import ngrok
!streamlit run test.py&>/dev/null&
p_url = ngrok.connect(addr='8501')
p_url
```

4. After using the project, you have to kill steamlit and ngrok. In the list of '!ps', find the PID of streamlit and kill the PID.
```python
!ps
!kill PID_steamlit
!ps
```

5. Also kill ngrok and check it.
```python
ngrok.kill()
!ps
```


# Tests
https://user-images.githubusercontent.com/48444301/173996103-154ff15c-2b5b-4b8f-ab7c-3aa1e34b1816.mp4

