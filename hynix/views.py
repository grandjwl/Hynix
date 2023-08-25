from django.db import models as dj_models
from io import StringIO
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Prediction
import torch
import shutil
import re
import os
import csv
import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score
import pickle
from pycaret.regression import *
from sklearn.utils import resample
import pandas as pd
import numpy as np
import json
from django.http import JsonResponse
import torch.nn as nn
from hynix.ensemble import DataPreprocessor, PreprocessAndPredict, Preprocessor, LSTM
import random
from hynix.ensemble import LSTM_model

# 50번의 시뮬레이션 데이터 생성
def MakeSimulationData(test, filename):
    prediction_obj = Prediction.objects.filter(csv_file__contains=filename).first()

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
    # test_data = MakeSimulationData(test_data, 'prepro_kdy.csv')
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
    # test_data = MakeSimulationData(test_data, 'prepro_cgw.csv')
    pp = PreprocessAndPredict()
    pred = pp.run(test_data)
    # 100 곱하기
    pred = pd.DataFrame(pred.detach().numpy())
    return pred

# 정우
def model3_prediction(test_data):
    # test_data = MakeSimulationData(test_data, 'prepro_ljw.csv')
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
    mse_values = [10.74511, 11.8428, 1.9966] # 동연, 고운, 정우
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

def calculate_confidence_interval(predictions, alpha=0.9):
    # Bootstrap re-sampling을 사용하여 신뢰구간 계산
    bootstrapped_samples = [resample(predictions) for _ in range(1000)]
    min = np.percentile(bootstrapped_samples, (1-alpha)/2*100)
    max = np.percentile(bootstrapped_samples, alpha+((1-alpha)/2)*100)
    return min, max

def simulation(request):
    prediction = None
    confidence_interval = {"min": None, "max": None, "mean": None}
    test_data_json = '{}'  

    if request.method == "POST":
        if 'test_data' in request.FILES:
            test_data_file = request.FILES['test_data']
            
            test_data = pd.read_csv(test_data_file)
            
            try:
                # 'unnamed 0' 컬럼 제거
                test_data = test_data.drop(columns=['Unnamed: 0'], errors='ignore')
                test_data.to_csv('hynix/read_csv/test_data.csv')
                
                # ID 컬럼을 key로 하는 json 생성
                test_data_json = test_data.head(10).set_index('ID').to_json(orient='index')
            except Exception as e:
                test_data_json = '{}'
                print(f"Error converting data to JSON: {e}")
        
            # prediction = ensemble_models(test_data_file)
            prediction = np.random.randint(10,100,833)
            min_val = [min(prediction) for i in range(10)]
            max_val = [max(prediction) for i in range(10)]
            mean_val = [sum(prediction)//len(prediction) for i in range(10)]
            
            # 신뢰구간 계산
            # min, max = calculate_confidence_interval(prediction["prediction"])
            # min_val, max_val, mean_val = calculate_confidence_interval(prediction)
            confidence_interval = {"min": min_val, "max": max_val, "mean": mean_val}

    return render(request, "hynix/simulation.html", {"prediction": prediction, "data_json": test_data_json, "confidence_interval": confidence_interval})

def lifecycle(request):
    # prediction 가져오기
    try:
        latest_prediction = Prediction.objects.latest('date_created')
        prediction_file = latest_prediction.csv_file.path
        
        # prediction CSV 파일을 df로 읽기
        prediction_data = pd.read_csv(prediction_file)
        
        # 필요한 컬럼을 추출하고 딕셔너리 리스트로 변환
        data = prediction_data[["Date", "ID", "prediction"]].to_dict(orient='records')
        
    except (Prediction.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        # 오류 발생 시, 기본 데이터를 생성해 data 리스트에 저장
        data = []
        for i in range(1, 101):
            datetime_value = "2023-08-23 00:00:00"
            lot_id = 1000 + i
            pred = 70 + i
            real = None
            data.append({"Date": datetime_value, "ID": lot_id, "Pred": pred, "Real": real})

    if request.method == "POST":
        # 사용자가 입력한 'real' 데이터를 받아오기
        updated_data = request.POST.get('real_values', [])  # 'real_values'라는 키로 데이터가 전송되지 않았을 때 기본 값으로 설정되는 값
        
        # POST 요청이 오는 경우에만 prediction_data를 다시 가져와야 함
        # prediction_data 초기화
        try:
            latest_prediction = Prediction.objects.latest('date_created')
            prediction_file = latest_prediction.csv_file.path
            prediction_data = pd.read_csv(prediction_file)
        except (Prediction.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            # 오류 발생 시, 기본 데이터를 생성해 prediction_data를 초기화
            prediction_data = pd.DataFrame()
            prediction_data = prediction_data.append(data, ignore_index=True)
            latest_prediction = Prediction()        
        
        # 'real' 컬럼 데이터 추가
        for item in updated_data:
            prediction_data.loc[prediction_data['ID'] == item['ID'], 'real'] = item['real']
        
        # 델타 값 계산
        prediction_data['delta'] = (prediction_data['prediction'] - prediction_data['real']).abs()
        
        # 델타 값의 평균 계산
        avg_delta = prediction_data['delta'].mean()
        prediction_data['avg_del'] = avg_delta
        Date = prediction_data['Date'] 
        
        # 데이터를 DB에 다시 저장
        prediction_csv = StringIO()
        prediction_data.to_csv(prediction_csv, index=False)
        latest_prediction.csv_file = ContentFile(prediction_csv.getvalue().encode('utf-8'), name="updated_predictions.csv")
        latest_prediction.save()
        

    # 데이터와 함께 lifecycle.html 페이지를 렌더링해 화면에 보여줌
    return render(request, "hynix/lifecycle.html", {"data": data})