# Tacademy ASAC 2기 하이주니어 팀
## SK 하이닉스 기업연계 프로젝트 
## 반도체 양산 공정 데이터를 활용한 수율 시뮬레이터 개발 

<hr>

### INDEX 
1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [프로젝트 환경](#프로젝트-환경)
4. [데이터와 모델](#데이터와-모델)
5. [참고문헌](#참고-문헌)

<hr>

### 프로젝트 개요 
-  최근 SK하이닉스 고객사들의 개발 TAT 단축 요청이 지속적으로 증가 하고 있는 상황 입니다. TAT란 서비스 과정에서 어떤 작업이 시작되어 완료될 때까지의 시간을 의미합니다. 그 예시로 SK하이닉스 고객사인 NVIDIA가 그래픽카드 출시일을 앞당긴 사례도 있습니다. 이러한 TAT를 맞추기 위해서 같은 시간 내에 더 많은 제품들을 안정성 있게 생산하여 고객사에게 제공해야 합니다. 그래서 저희는 미완료된 공정의 수율을 예측하여, 최종 수율을 올리고 더 많은 정상품을 생산 해야겠다고 생각 했습니다. 
  
-  우선 반도체 공정은 Fab이라는 대규모 생산 시설에서 칩으로 제조됩니다. 복잡한 프로세스지만 크게 전공정 그리고 후공정으로 나뉩니다. 먼저 전공정에서 웨이퍼 투입(Fab in)부터 여러 공정들을 거쳐 박막공정까지 마친 후(Fab out), 후공정에서 테스트 및 패키징 공정까지 이뤄지게 됩니다. 저희는 이러한 전공정에서 추가로 계측공정을 진행해서 쌓인 데이터를 모델링에 사용해 수율을 예측하도록 하였습니다.
<br><br>

### 프로젝트 구조 
![image](https://github.com/grandjwl/Hynix/assets/135038257/36ba5f3f-c92e-45d9-8918-070ba4786684)
>저희 프로젝트 구조입니다. 
- 반도체 공정에서 발생하는 센서값들을 받아 수율 예측 모델을 돌리고 웹 페이지에서 이를 가시화 합니다. 그리고 재학습의 기준을 확인하여 모델 라이프사이클을 체크 할 수 있도록 기획 했습니다.
<br><br>

### 프로젝트 환경
![image](https://github.com/grandjwl/Hynix/assets/135038257/a97e4dd1-d7f8-4654-8b35-ed7a01aea436)
>저희 프로젝트 환경입니다.
- 머신러닝에서는 scikit-learn과 Pycaret을 사용 했고 딥러닝에서는 PyTorch를 사용 했으며 웹에서는 django 기반 프레임워크를 사용 후 UI개발은 Boostrap을 사용 했습니다. 
<br><br>

### 데이터와 모델
<br><br>

### 참고 문헌
- [SK 하이닉스 기술블로그 (블로그)](https://news.skhynix.co.kr/post/jeonginseong-column-computer)

- [점프 투 장고 (E-book)](https://wikidocs.net/book/4223)

- Django로 배우는 쉽고 빠른 웹 개발_파이썬 웹프로그래밍 (도서)

- Liu, P., Sun, X., Han, Y., He, Z., Zhang, W., Wu, C. (2022) Arrhythmia classification of LSTM autoencoder based on
time series anomaly detection. Biomedical Signal Processing and Control

- Xu, X., Yoneda, M. (2019) Multitask air-quality prediction based on LSTM-autoencoder model. IEEE transactions
on cybernetics 5

- 카이스트 주재걸 교수 LSTM 강의 (동영상)
<br><br>












