from django.db import models as dj_models
import os

def user_directory_path(instance, filename):
    # 파일 확장자 추출
    ext = filename.split('.')[-1]
    # 파일 이름을 instance의 ID와 확장자로 지정 (ID가 없는 경우, 기본적으로 filename을 사용)
    filename = f"{instance.ID if instance.ID else filename}.{ext}"
    return os.path.join('predictions/', filename)

# 2. 셋의 전처리만 완료된 train.csv가 저장된 테이블
class PreprocessedCSV(dj_models.Model):
    name = dj_models.CharField(max_length=100)
    data = dj_models.FileField(upload_to='input_csvs/')
    
    def __str__(self):
        return self.name
    
# 3. 앙상블 완료된 'test.csv+앙상블prediction' df를 csv로 저장하는 테이블
class Prediction_complete(dj_models.Model):
    name = dj_models.CharField(max_length=100)
    csv_file = dj_models.FileField(upload_to=user_directory_path)

# 4. simulation 페이지에서 필요한거 저장 및 사용하는 테이블    
class Wsimulation(dj_models.Model):
    test_csv = dj_models.FileField(upload_to='input_csvs/') 
    min_value = dj_models.FloatField(null=True, blank=True)  
    max_value = dj_models.FloatField(null=True, blank=True)
    avg_value = dj_models.FloatField(null=True, blank=True)  # 평균값 저장을 위한 필드 추가
    # last_columns = 공정의 마지막 컬럼명 저장 후 프런트로 보내기
    

# 5. lifecycle 페이지에서 필요한거 저장 및 사용하는 테이블 
class WLifecycle(dj_models.Model): # lifecycle은 맨아래 행만 필요  
    test_csv = dj_models.FileField(upload_to='input_csvs/') # 업로드된 test.csv 원본
    Lot_ID = dj_models.CharField(max_length=100)
    avg_value = dj_models.FloatField(null=True, blank=True)  # 평균값 저장을 위한 필드 추가
    real = dj_models.FloatField(null=True, blank=True)
    real_input_time = dj_models.DateTimeField(auto_now_add=True)