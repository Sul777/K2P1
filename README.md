# 2021 Pusan National University CSE Graduation Project

- team name : K2P1
- project name : Portrait rights protection system


# How to use?

구글 코랩으로 이 프로젝트 파일을 불러오세요.
Open the file named '2021-PNU-K2P1-Portrait rights protection system.ipynb' on Google Colab. 

1. Install libraries.
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
```

2. Restart the Google Colab runtime and then run the following code.
Select Runtime > Restart runtime
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Run STREAMLIT code after the upper code.

4. Sign up [ngrok](https://ngrok.com/) and get your authtoken. Then write and execute your authtoken after '!ngrok'.
```python
!ngrok your_authtoken
```

5. Execute the below codes.
```python
from pyngrok import ngrok
!streamlit run test.py&>/dev/null&
p_url = ngrok.connect(addr='8501')
p_url
```

6. After using the project, you have to kill steamlit and ngrok. In the list of '!ps', find the PID of streamlit and kill the PID.
```python
!ps
!kill PID_steamlit
```

7. Also kill ngrok.
```python
ngrok.kill()
```


# Tests
