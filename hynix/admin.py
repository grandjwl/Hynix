from django.contrib import admin
from .models import PreprocessedCSV, Prediction_complete #Wsimulation, WLifecycle

# 각 모델을 관리자 사이트에 등록합니다.
admin.site.register(PreprocessedCSV)
admin.site.register(Prediction_complete)
# admin.site.register(Wsimulation)
# admin.site.register(WLifecycle)