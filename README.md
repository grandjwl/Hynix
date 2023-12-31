# Tacademy ASAC 2기 하이주니어 팀
## SK 하이닉스 기업연계 프로젝트 
## 반도체 양산 공정 데이터를 활용한 수율 시뮬레이터 개발 

<hr>

### INDEX 
1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [프로젝트 환경](#프로젝트-환경)
4. [데이터와 모델](#데이터와-모델)
5. [웹 시뮬레이터 개발](#웹-시뮬레이터-개발)
6. [추후 연구 방향](#추후-연구-방향)
7. [참고문헌](#참고-문헌)

<hr>

### 프로젝트 개요 
-  최근 SK하이닉스 고객사들의 개발 TAT 단축 요청이 지속적으로 증가 하고 있는 상황 입니다. 그 예시로 SK하이닉스 고객사인 NVIDIA가 그래픽카드 출시일을 앞당긴 사례가 있습니다. 이러한 TAT를 맞추기 위해서는 같은 시간 내에 더 많은 제품들을 안정성 있게 생산하여 고객사에게 제공해야 합니다. 따라서, 미완료된 공정의 수율을 예측하여 최종 수율을 올리고 더 많은 정상품을 생산함으로써 해당 문제를 극복할 수 있다고 판단하였습니다. 
   > TAT란 서비스 과정에서 어떤 작업이 시작되어 완료될 때까지의 시간을 의미합니다
  
-  반도체 공정은 Fab이라는 대규모 생산 시설에서 칩으로 제조되며 크게 전공정 그리고 후공정으로 나뉩니다. 먼저 전공정에서 웨이퍼 투입(Fab in)부터 여러 공정들을 거쳐 박막공정까지 마친 후(Fab out), 후공정에서 테스트 및 패키징 공정까지 이뤄지게 됩니다. 저희는 이러한 전공정에서 추가로 계측공정을 진행해서 쌓인 데이터를 모델링에 사용해 수율을 예측하였습니다.
<br><br>

### 프로젝트 구조 
![image](https://github.com/grandjwl/Hynix/assets/135038257/cb60bea1-12ea-452e-967c-4aeb3aa46ebc)
- 반도체 공정에서 발생하는 센서값들을 받아 수율 예측 모델을 돌리고 웹 페이지에서 이를 가시화 합니다. 그리고 재학습의 기준을 확인하여 모델 라이프사이클을 체크 할 수 있도록 기획 했습니다.
<br><br>

### 프로젝트 환경
![image](https://github.com/grandjwl/Hynix/assets/135038257/5bad9f3d-c4fd-47f7-967d-a0282a62405a)
- 머신러닝 모델링은  scikit-learn과 Pycaret을 사용 했고 딥러닝 모델링은 PyTorch를 사용 했습니다.
- 웹 백엔드 개발 프레임워크로 django를 사용했으며 웹 프론트엔드 개발은 Boostrap과 HTML, CSS, JS를 활용했습니다. 
<br><br>

### 데이터와 모델
- Data
   - 반도체 공정에서 발생하는 2496 * 1484 형태의 비식별화된 데이터셋을 SK 하이닉스로부터 제공받아 사용하였습니다.
   - 시계열 데이터의 특성을 가집니다.
- Model
   - 해당 프로젝트의 최종 수율 예측 모델은 Pycaret을 활용하여 선정한 5가지의 베이스 모델을 앙상블한 ML 모델입니다.
      > 모델링에 대한 자세한 내용은 발표자료를 참고부탁드립니다.

<br><br>

### 웹 시뮬레이터
#### 개발 목적 및 이점
-  반도체 제조 과정에서의 수율은 매우 중요합니다. 높은 수율은 생산 효율성과 수익을 증가시킵니다. 그래서 수율 예측은 사업 전략과 생산 프로세스를 개선하는 데 큰 역할을 합니다. 웹 시뮬레이터는 이를 위해 사내 반도체 수율 예측 도구로 활용될 수 있습니다. 이뿐만 아니라, 공정 내 다양한 이슈를 탐지할 수 있으며, 수율 예측 결과와 공정 트렌드에 관한 시각화 자료도 제공합니다. 이로써 업무 효율을 높이고, 인사이트를 제공할 수 있습니다.

#### 시뮬레이션 방식
-  먼저, 행이 1개만 있는 미완료 공정 데이터가 입력되면, 이를 기반으로 100개의 복제 데이터를 만듭니다. 이 복제 과정에서 모델 학습 시 사용한 Train 데이터의 각 컬럼에서 랜덤으로 선택된 값으로 Null 값을 채웁니다. 이렇게 만들어진 복제 데이터는 Train 데이터 전처리 기준으로 처리됩니다. 시뮬레이션 후에는 예측값 100개 중에서 90% 신뢰구간의 Min, Max 범위와 그 평균값이 제공됩니다.

#### Requirements
  1. 개발 환경 세팅
      ```
      conda create -n hynix python=3.10.12
      conda activate hynix
      pip install -r requirements.txt
      ```

     
  2. 웹 서버 시작하기
  
     ```
     python manage.py runserver
     ```

#### Simulation
![image](https://github.com/grandjwl/Hynix/assets/127659652/aff9e63b-8732-4cd7-905a-831ea00452c5)
   - 웹 시뮬레이터의 Simulation 페이지 입니다. 웹 시뮬레이터의 공정이 완료되기 전의 데이터와 완료된 데이터 모두 업로드하여 수율 예측값을 시뮬레이션 가능합니다.
<br><br>

![image](https://github.com/grandjwl/Hynix/assets/127659652/5bfcb41e-b829-45d3-ad2b-d94863432ecb)  
   - 공정의 완료 여부와 상관없이 데이터를 업로드하면 업로드한 데이터를 웹에서 확인 가능합니다.
   - 'SHOW CHART' 버튼을 눌러 공정이 완료되기 전부터 시뮬레이터를 통해 수율을 예측할 수 있습니다.
<br><br>
  
![image](https://github.com/grandjwl/Hynix/assets/127659652/b473caad-f70e-4972-bd3d-83457cef292c)
   - 최종적으로 공정이 완료된 데이터를 시뮬레이션하여 공정이 완료될때까지의 시뮬레이션 결과를 그래프를 통해 확인할 수 있습니다.
   - 파란색 선과 노란색 선은 예측한 수율 값의 신뢰구간이고, 초록색 점들은 예측 값들의 평균 값입니다.
<br><br>

#### Model Life Cycle
![image](https://github.com/grandjwl/Hynix/assets/127659652/df467b99-55ed-4c03-b800-76822c33af47)
   - 지금까지 예측했던 공정이 완료된 데이터를 불러옵니다. 화면 내 표 최하단에는 마지막에 입력한 데이터가 들어있습니다.
<br><br>

![image](https://github.com/grandjwl/Hynix/assets/127659652/19927df3-ca01-47ee-b02e-5b3f34a93a75)
   - 표 최하단 행의 real 컬럼값으로 마지막에 입력한 데이터에 대한 실제 수율 값을 입력해줍니다.
<br><br>

![image](https://github.com/grandjwl/Hynix/assets/127659652/2c1e1329-bf99-4139-9254-fbc0aaab8213)
   - 'UPDATE DATA' 버튼을 눌러 실제 수율값과 예측된 수율값의 오차를 입력 순서대로 그래프를 통해 확인 가능합니다.
   - 그래프의 값이 설정한 Threshold를 넘으면 재학습을 고려해 볼 수 있습니다.
<br><br>

### 추후 연구 방향  
- 공정 진행 단계에서 발생할 수 있는 매우 많은 변수들과 공정들의 특성 상 데이터가 수율과 직접적인 상관성을 가지기 어렵기 때문에 예측 불가능한 변수들에 대해 세밀하게 모델링을 진행한다면 예측력을 높이고 현업 적용까지 기대해볼 수 있습니다.
- 시계열 데이터의 특성을 고려한 더욱 다양한 전처리 방식을 도입하여 성능 개선을 시도할 수 있습니다.
- 더욱 방대한 양의 데이터를 확보하여 모델의 일반화 능력을 개선할 수 있습니다.
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












