from django.contrib import admin
from .models import InputCSV, PreprocessedCSV, WLifecycle, Prediction_complete

# 각 모델을 관리자 사이트에 등록합니다.
admin.site.register(InputCSV)
admin.site.register(PreprocessedCSV)
admin.site.register(WLifecycle)
admin.site.register(Prediction_complete)