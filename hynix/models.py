from django.db import models as dj_models
import os

def user_directory_path(instance, filename):
    # 파일 확장자 추출
    ext = filename.split('.')[-1]
    # 파일 이름을 instance의 ID와 확장자로 지정 (ID가 없는 경우, 기본적으로 filename을 사용)
    filename = f"{instance.id if instance.id else filename}.{ext}"
    return os.path.join('predictions/', filename)

class Prediction_complete(dj_models.Model):
    name = dj_models.CharField(max_length=100) # 최대 100자까지의 문자열 데이터를 저장
    csv_file = dj_models.FileField(upload_to=user_directory_path)

class InputCSV(dj_models.Model):
    name = dj_models.CharField(max_length=100) # 최대 100자까지의 문자열 데이터를 저장
    data = dj_models.FileField(upload_to='input_csvs/')
    
class PreprocessedCSV(dj_models.Model):
    name = dj_models.CharField(max_length=100)  # 최대 100자까지의 문자열 데이터를 저장
    data = dj_models.FileField(upload_to='input_csvs/')
    
    def __str__(self):
        return self.name
    
class WLifecycle(dj_models.Model):
    avg_value = dj_models.FloatField(null=True, blank=True)  # 평균값 저장을 위한 필드 추가
    min_value = dj_models.FloatField(null=True, blank=True)  
    max_value = dj_models.FloatField(null=True, blank=True)  
    date_created = dj_models.DateTimeField(auto_now_add=True)