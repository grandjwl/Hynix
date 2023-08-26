from django.db import models as dj_models
from io import StringIO
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import PreprocessedCSV, Prediction_complete, InputCSV, WLifecycle
import pandas as pd
import numpy as np
import shutil
import re
import os
import csv
import io
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore')
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score
import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.utils import resample
from hynix.ensemble import DataPreprocessor, PreprocessAndPredict, Preprocessor, LSTM
import random
from hynix.ensemble import LSTM_model
from pycaret.regression import *
import json

# 50번의 시뮬레이션 데이터 생성
def MakeSimulationData(test, filename):
    prediction_obj = PreprocessedCSV.objects.filter(data__contains=filename).first()

    if prediction_obj:
        file_path = prediction_obj.csv_file.path
        
        # 파일을 직접 Pandas 데이터프레임으로 읽기
        train_df = pd.read_csv(file_path)

    # test2의 행을 50번 복제
    test = pd.concat([test]*50).reset_index(drop=True)

    # 각 행에 대해 처리
    for i in range(len(test)):
        # 해당 행에서 마지막으로 값이 있는 컬럼 찾기
        last_valid_col = test.iloc[i].last_valid_index()

        # last_valid_col 다음 컬럼부터 값을 채우기
        for col in test.columns[test.columns.get_loc(last_valid_col)+1:]:
            # 해당 컬럼에서 랜덤한 값을 선택
            random_value = random.choice(train_df[col].tolist())
            test[col].iloc[i] = random_value
    return test

# 동연
def model1_prediction(test_data):
    test_data = MakeSimulationData(test_data, 'prepro_kdy.csv')
    dp = DataPreprocessor()
    deleted_columns, null_columns, to_drop_all = dp.load_pickles()
    final_test = dp.preprocessing_realtest(test_data)
    scaled_final_test = dp.scale_data_without_target(final_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('hynix/ensemble/kdy/final_best_model.pt')
    model.eval()

    test_tensor = torch.tensor(scaled_final_test.values).float().to(device)
    test_tensor = test_tensor.view(-1, 1, 1285)

    with torch.no_grad():
        predictions = model(test_tensor)

    predictions_test = predictions.cpu().numpy()
    predictions_test = predictions_test * 100

    pred = pd.DataFrame(predictions_test) # 833 rows × 1 columns
    return pred

# 고운
def model2_prediction(test_data):
    test_data = test_data.iloc[:, 1:]
    test_data = MakeSimulationData(test_data, 'prepro_cgw.csv')
    pp = PreprocessAndPredict()
    pred = pp.run(test_data)
    # 100 곱하기
    pred = pd.DataFrame(pred.detach().numpy())
    return pred

# 정우
def model3_prediction(test_data):
    test_data = test_data.iloc[:, 1:]
    test_data = MakeSimulationData(test_data, 'prepro_ljw.csv')
    # default_path = 'hynix/ensemble/ljw/'
    # with open(default_path + "Preprocessor", "rb") as f:
    #     preprocess = pickle.load(f)
    ps = Preprocessor()
    Rtest = ps.preprocessing_Rtest(test_data)

    best_model = load_model('hynix/ensemble/ljw/ML_best_model')
    pred = predict_model(best_model, data=Rtest)['prediction_label']*100
    pred = pd.DataFrame(pred)
    return pred


def ensemble_models(test_data):
    mse_values = [10.73052, 11.8428, 1.9966] # 동연, 고운, 정우
    weights = [1/mse for mse in mse_values]
    normalized_weights = [weight/sum(weights) for weight in weights]
    
    pred1 = model1_prediction(test_data)
    pred2 = model2_prediction(test_data)
    pred3 = model3_prediction(test_data)
    
    pred1 = pred1.reset_index(drop=True)
    pred2 = pred2.reset_index(drop=True)
    pred3 = pred3.reset_index(drop=True)

    ensemble_module = pred1.iloc[:, 0]*normalized_weights[0] + pred2.iloc[:, 0]*normalized_weights[1] + pred3.iloc[:, 0]*normalized_weights[2]
    predictions = ensemble_module
    predictions.columns = ['prediction']
    prediction = pd.concat([test_data, predictions], axis=1)
    return prediction

def calculate_confidence_interval(predictions_df, alpha=0.9):
    # predictions = predictions_df['prediction']
    predictions = predictions_df

    # Bootstrap re-sampling을 사용하여 신뢰구간 계산
    bootstrapped_samples = [resample(predictions) for _ in range(1000)]
    min_val = np.percentile(bootstrapped_samples, (1-alpha)/2*100)
    max_val = np.percentile(bootstrapped_samples, alpha+((1-alpha)/2)*100)
    mean_val = predictions.mean()

    return min_val, max_val, mean_val

def main(request):
    return render(request, 'hynix/main.html',{"contents":""})

from sklearn.utils import resample

def calculate_confidence_interval(predictions_df, alpha=0.9):
    predictions = predictions_df['prediction']
    # predictions = predictions_df

    # Bootstrap re-sampling을 사용하여 신뢰구간 계산
    bootstrapped_samples = [resample(predictions) for _ in range(1000)]
    min_val = np.percentile(bootstrapped_samples, (1-alpha)/2*100)
    max_val = np.percentile(bootstrapped_samples, alpha+((1-alpha)/2)*100)
    mean_val = predictions.mean()

    return min_val, max_val, mean_val

def simulation(request):
    prediction = None
    confidence_interval = {"min": None, "max": None, "mean": None}
    test_data_json = '{}'  
    test_data = None
    last_filled_column_name = None

    if request.method == "POST" and 'test_data' in request.FILES:
        test_data_file = request.FILES['test_data']
        input_csv_instance = InputCSV(name=test_data_file.name, data=test_data_file)
        input_csv_instance.save()
        test_data = pd.read_csv(test_data_file)
        test_data = test_data.iloc[0,:]
        test_data = test_data.to_frame().transpose()
        
        # 첫 번째 행에서 마지막으로 값이 있는 컬럼의 이름을 찾기
        first_row = test_data.iloc[0].dropna()
        if not first_row.empty:
            last_filled_column_name = first_row.index[-1]
        
        # test_data를 JSON 형태로 변환
        test_data_temp = test_data
        test_data_temp = test_data_temp.drop(columns=['Unnamed: 0'], errors='ignore')
        test_data_json = test_data_temp.to_json(orient='records')
    
        prediction = ensemble_models(test_data)
        # 신뢰구간 계산
        min_val, max_val, mean_val = calculate_confidence_interval(prediction)
        
        # prediction = np.random.randint(10,100,50)
        # min_val = [min(prediction) for i in range(10)]
        # max_val = [max(prediction) for i in range(10)]
        # mean_val = [sum(prediction)//len(prediction) for i in range(10)]
        # min_val, max_val, mean_val = calculate_confidence_interval(prediction)
        confidence_interval = {"min": min_val, "max": max_val, "mean": mean_val}
        
        lifecycle_instance = WLifecycle(min_value=min_val, max_value=max_val)
        lifecycle_instance.save()
        
        # prediction_csv = StringIO()
        # prediction.to_csv(prediction_csv, index=False)
        
        # 예측 결과를 Prediction_complete 모델에만 저장
        # prediction_complete_model = Prediction_complete(name="Prediction Result", csv_file=ContentFile(prediction_csv.getvalue().encode('utf-8'), name="complete_predictions.csv"))
        # prediction_complete_model.save()
    return render(request, "hynix/simulation.html", {"prediction": prediction, "data_json": test_data_json, "confidence_interval": confidence_interval})

def lifecycle(request):
    # prediction 가져오기
    try:
        latest_prediction = Prediction_complete.objects.latest('id')  # 'id'는 Django의 기본 제공 필드로, 최신 데이터를 가져오는 데 사용됩니다.
        prediction_file = latest_prediction.csv_file.path
        
        # prediction CSV 파일을 df로 읽기
        prediction_data = pd.read_csv(prediction_file)
        
        # 필요한 컬럼을 추출하고 딕셔너리 리스트로 변환
        data = prediction_data[["Date", "ID", "prediction"]].to_dict(orient='records')
        
    except (Prediction_complete.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        # 오류 발생 시, 기본 데이터를 생성해 data 리스트에 저장
        data = []
        for i in range(1, 101):
            datetime_value = "2023-08-23 00:00:00"
            lot_id = 1000 + i
            pred = 70 + i
            real = 70
            data.append({"Date": datetime_value, "ID": lot_id, "prediction": pred, "real": real})



    if request.method == "POST":

        # 사용자가 입력한 'IDReal' 데이터를 JSON 형태로 받아오기
        updated_data_json = request.POST['IDreal'] 
        # JSON 데이터를 딕셔너리로 변환
        updated_data_dict = json.loads(updated_data_json)
        print(updated_data_dict)
        # POST 요청이 오는 경우에만 prediction_data를 다시 가져와야 함
        # prediction_data 초기화
        try:
            latest_prediction = Prediction_complete.objects.latest('id')
            prediction_file = latest_prediction.csv_file.path
            prediction_data = pd.read_csv(prediction_file)
        except (Prediction_complete.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            # 오류 발생 시, 기본 데이터를 생성해 prediction_data를 초기화
            prediction_data = pd.DataFrame()
            prediction_data = prediction_data.append(data, ignore_index=True)
        
        for item in updated_data_dict:
            id_value = item["ID"]
            real_value = item['real']   
            # prediction_data에서 'ID' 값이 일치하는 행을 찾아서 'real' 값을 대체
            prediction_data.loc[prediction_data['ID'] == id_value, 'real'] = real_value
        
        # 델타 값 계산
        prediction_data['delta'] = (prediction_data['prediction'] - prediction_data['real']).abs()
        
        # 델타 값의 평균 계산
        avg_delta = prediction_data['delta'].mean()
        prediction_data['avg_del'] = avg_delta
        Date = prediction_data['Date'][0] 
        print(avg_delta)
        print(Date)
        
        # 데이터를 DB에 다시 저장
        context = {"avg_delta": avg_delta, "Date": Date}
        return render(request, "hynix/lifecycle.html", context)

    # POST 요청이 아닌 경우에도 데이터를 전달할 context 생성
    context = {"data": data}

    # 데이터와 함께 lifecycle.html 페이지를 렌더링해 화면에 보여줌
    return render(request, "hynix/lifecycle.html", context)