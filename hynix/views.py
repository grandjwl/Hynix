from .models import PreprocessedCSV, Prediction_complete, WLifecycle, Wsimulation
from django.db import models as dj_models
from io import StringIO
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
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
from hynix.ensemble import DataPreprocessor, PreprocessAndPredict, Preprocessor
import random
from pycaret.regression import *
import json
import pandas as pd
from sklearn.utils import resample
from hynix.model_class import LSTM, LSTM_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 50번의 시뮬레이션 데이터 생성
def MakeSimulationData(test, filename):
    prediction_obj = PreprocessedCSV.objects.filter(data__contains=filename).first()

    if prediction_obj:
        file_path = prediction_obj.data.path
        
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
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            out, _ = self.lstm1(x, (h0, c0))
            out = self.dropout(out)
            out, _ = self.lstm2(out)
            out = self.fc(out[:, -1, :])
            return out
    class DataPreprocessor:
        def __init__(self):
            self.deleted_columns = None
            self.null_columns = None
            self.to_drop_all = None

        def sort_time(self, data):
            for idx,col in enumerate(data.columns[188:1454]):
                if data[col].dtype == "object":
                    data[col] = pd.to_datetime(data[col])
            ts_data = data.select_dtypes("datetime")

            for idx in ts_data.index:
                ts_data.sort_values(by=idx,axis=1, inplace=True)
            result = []
            datatmp = data.columns.to_list()[188:1454]
            for idx,col in enumerate(datatmp):
                if data[col].dtype == "<M8[ns]":
                    cur = int(col[1:]) # x195 -> 195
                    i = idx
                    tmp = []
                    while i > 0:
                        i -= 1
                        next = datatmp[i] # x194
                        if data[next].dtype == "<M8[ns]":
                            break
                        else:
                            tmp.append(next)
                            tmp.sort()
                    result.append((col,tmp))
            ts_final = []
            for elem in ts_data.columns:
                for target,content in result:
                    if elem == target:
                        ts_final.extend(content)
                        ts_final.append(target)
            ts_final = data[ts_final]
            front = data.loc[:,:"x193"]
            back = data.loc[:,"x1461":]
            data = pd.concat([front, ts_final, back], axis = 1)
            return data

        def Qtime(self, data, ts_data):
            df = pd.DataFrame(index=data.index)
            for idx in range(1, len(ts_data.columns)):
                col = []
                for jdx in range(len(ts_data.index)):
                    time1 = datetime.strptime(ts_data.iloc[jdx,idx],"%Y-%m-%d %H:%M:%S")
                    time2 = datetime.strptime(ts_data.iloc[jdx,idx-1],"%Y-%m-%d %H:%M:%S")
                    diff =  time1 - time2
                    col.append(round(diff.seconds/(60*60),2))
                df[ts_data.columns[idx]+"-"+ts_data.columns[idx-1]] = col
            with open('hynix/ensemble/kdy/train_q_train.pkl', 'wb') as f:
                pickle.dump(df, f)
            return df

        def delete_null1(self, final):
            empty_columns = final.columns[final.isnull().all()]
            final = final.drop(empty_columns, axis=1)
            self.deleted_columns = list(empty_columns)
            with open('hynix/ensemble/kdy/deleted_columns.pkl', 'wb') as f:
                pickle.dump(self.deleted_columns, f)
            return final, self.deleted_columns

        def delete_null2(self, final):
            null_threshold = 0.95 
            null_counts = final.isnull().sum() 
            total_rows = final.shape[0]
            self.null_columns = null_counts[null_counts / total_rows >= null_threshold].index.tolist()
            final = final.drop(self.null_columns, axis=1)
            with open('hynix/ensemble/kdy/null_columns.pkl', 'wb') as f:
                pickle.dump(self.null_columns, f)    
            return final, self.null_columns
        
        def fillna_null(self, final):
            final.drop(final.columns[0], axis=1, inplace = True)
            null_columns = final.columns[final.isnull().any()]
            for column in null_columns:
                noise = np.random.normal(loc=0, scale=0.01, size=final[column].isnull().sum())
                final.loc[final[column].isnull(), column] = noise
            return final

        def drop_corrfeature(self, final):
            corr_matrix_all = final.corr().abs()
            upper_all = corr_matrix_all.where(np.triu(np.ones(corr_matrix_all.shape), k=1).astype(bool))
            to_drop_all = [column for column in upper_all.columns if any(upper_all[column] > 0.8)]
            self.to_drop_all = [column for column in upper_all.columns if any(upper_all[column] > 0.8)]
            final = final.drop(self.to_drop_all, axis=1)
            with open('hynix/ensemble/kdy/to_drop_all.pkl', 'wb') as f:
                pickle.dump(self.to_drop_all, f) 
            return final, self.to_drop_all
        
        def preprocessing_train(self, data):
            a = self.sort_time(data)
            train_ts_data = a.select_dtypes("datetime").astype("str")
            train_q = self.Qtime(a,train_ts_data)
            final, deleted_columns = self.delete_null1(data)
            final, null_columns = self.delete_null2(final)
            final = self.fillna_null(final)
            final, to_drop_all = self.drop_corrfeature(final)
            final = final.sort_index()
            final =  final.select_dtypes(include=['float64'])
            final = pd.concat([final,train_q],axis=1)
            return final, to_drop_all, deleted_columns, null_columns
        
        def preprocessing_test(self, data):
            a = self.sort_time(data)
            train_ts_data = a.select_dtypes("datetime").astype("str")
            train_q = self.Qtime(a,train_ts_data)
            final = data.drop(self.deleted_columns, axis=1)
            final = final.drop(self.null_columns, axis=1)
            final = self.fillna_null(final)
            final = final.drop(self.to_drop_all, axis=1)
            final = final.sort_index()
            final =  final.select_dtypes(include=['float64'])
            final = pd.concat([final,train_q],axis=1)
            return final
        
        def load_pickles(self):
            with open('hynix/ensemble/kdy/deleted_columns.pkl', "rb") as file:
                deleted_columns = pickle.load(file)
            with open('hynix/ensemble/kdy/null_columns.pkl', "rb") as file:
                null_columns = pickle.load(file)
            with open('hynix/ensemble/kdy/to_drop_all.pkl', "rb") as file:
                to_drop_all = pickle.load(file)
            return deleted_columns, null_columns, to_drop_all
        
        def preprocessing_realtest(self, data):
            deleted_columns, null_columns, to_drop_all = DataPreprocessor.load_pickles(self)
            a = self.sort_time(data)
            train_ts_data = a.select_dtypes("datetime").astype("str")
            train_q = self.Qtime(a,train_ts_data)
            final = data.drop(deleted_columns, axis=1)
            final = final.drop(null_columns, axis=1)
            final = self.fillna_null(final)
            final = final.drop(to_drop_all, axis=1)
            final = final.sort_index()
            final =  final.select_dtypes(include=['float64'])
            final = pd.concat([final,train_q],axis=1)
            return final
        
        def scale_data(self, temp):
            scaler = MinMaxScaler()
            final_temp = pd.DataFrame(temp['Y'])
            temp = temp.drop('Y', axis=1)
            data_preprocessed_scaled = scaler.fit_transform(temp)
            data_preprocessed_scaled = pd.DataFrame(data_preprocessed_scaled, columns=temp.columns[:], index=temp.index)
            data_preprocessed_scaled['Y'] = final_temp['Y'] / 100 
            return data_preprocessed_scaled
        
        def scale_data_without_target(self, temp):
            scaler = MinMaxScaler()
            data_preprocessed_scaled = scaler.fit_transform(temp)
            data_preprocessed_scaled = pd.DataFrame(data_preprocessed_scaled, columns=temp.columns[:], index=temp.index)
            return data_preprocessed_scaled

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
    
    # pred1 = model1_prediction(test_data)
    pred2 = model2_prediction(test_data)
    # pred3 = model3_prediction(test_data)
    
    # pred1 = pred1.reset_index(drop=True)
    pred2 = pred2.reset_index(drop=True)
    # pred3 = pred3.reset_index(drop=True)

    # ensemble_module = pred1.iloc[:, 0]*normalized_weights[0] + pred2.iloc[:, 0]*normalized_weights[1] + pred3.iloc[:, 0]*normalized_weights[2]
    predictions = pred2.iloc[:, 0]*normalized_weights[1]

    return predictions

def calculate_confidence_interval(predictions, alpha=0.9):
    # Bootstrap re-sampling을 사용하여 신뢰구간 계산
    bootstrapped_samples = [resample(predictions) for _ in range(1000)]
    min_val = np.percentile(bootstrapped_samples, (1-alpha)/2*100)
    max_val = np.percentile(bootstrapped_samples, alpha+((1-alpha)/2)*100)
    mean_val = predictions.mean()

    return min_val, max_val, mean_val

def main(request):
    return render(request, 'hynix/main.html',{"contents":""})

def simulation(request):
    prediction = None
    confidence_interval = {"min": [], "max": [], "avg": []}
    test_data_json = '{}'  
    test_data = None
    last_filled_column_name = None

    if request.method == "POST" and 'test_data' in request.FILES:
        test_data_file = request.FILES['test_data']
        isFull = request.POST["isFull"]

        input_csv_instance = Wsimulation(test_csv=test_data_file)
        input_csv_instance.save()
        
        # is_full_data 값이 True인 경우 WLifecycle 테이블에도 저장
        if isFull:
            lifecycle_instance = WLifecycle(Lot_ID="Some Value", test_csv=test_data_file)
            lifecycle_instance.save()
        
        uploaded_file_path = input_csv_instance.test_csv.path
        test_data = pd.read_csv(uploaded_file_path)
        
        # 첫 번째 행에서 마지막으로 값이 있는 컬럼의 이름을 찾기 (멘토님이랑 상의 : 변경 예정)
        first_row = test_data.iloc[0].dropna()
        if not first_row.empty:
            last_filled_column_name = first_row.index[-1]
        
        # test_data를 JSON 형태로 변환
        test_data_temp = test_data
        test_data_temp = test_data_temp.drop(columns=['Unnamed: 0'], errors='ignore')
        test_data_json = test_data_temp.to_json(orient='records')
        
        prediction = ensemble_models(test_data)
        print("ensemble complete")
        # 신뢰구간 계산
        min_val, max_val, mean_val = calculate_confidence_interval(prediction)
        print("confidence interval complete")
        # 결과를 리스트로 저장
        confidence_interval["min"].append(min_val)
        confidence_interval["max"].append(max_val)
        confidence_interval["avg"].append(mean_val)
        
        simulation_instance = Wsimulation(min_value=min_val, max_value=max_val, avg_value=mean_val)
        lifecycle_instance = WLifecycle(avg_value=mean_val)
        simulation_instance.save()
        lifecycle_instance.save()

        prediction_csv = StringIO()
        prediction.to_csv(prediction_csv, index=False)

        # 예측 결과를 Prediction_complete 모델에만 저장
        # prediction_complete_model = Prediction_complete(name="Prediction Result", csv_file=ContentFile(prediction_csv.getvalue().encode('utf-8'), name="complete_predictions.csv"))
        # prediction_complete_model.save()
        
    return render(request, "hynix/simulation.html", {"prediction": prediction, "data_json": test_data_json, "confidence_interval": confidence_interval, "last_column": last_filled_column_name})

def lifecycle(request):
    predictions = []
    latest_id = None
    avg_delta = None
    latest_real_input_time = None
    
    # prediction 가져오기
    try:
        # 가장 최근에 생성된 Prediction_complete 데이터를 가져옴
        latest_prediction = Prediction_complete.objects.latest('id')  # 'id'는 Django의 기본 제공 필드로, 최신 데이터를 가져오는 데 사용됩니다.
        prediction_file = latest_prediction.csv_file.path
        
        # prediction CSV 파일을 df로 읽기
        prediction_data = pd.read_csv(prediction_file)
        
        # 필요한 컬럼을 추출하고 딕셔너리 리스트로 변환
        data = prediction_data[["ID", "prediction"]].to_dict(orient='records')
        
    except (Prediction_complete.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        # 오류 발생 시, 기본 데이터를 생성해 data 리스트에 저장
        data = []
        for i in range(1, 101):
            # datetime_value = "2023-08-23 00:00:00"
            lot_id = 1000 + i
            pred = 70 + i
            real = None
            data.append({"ID": lot_id, "prediction": pred, "real": real}) # "Date": datetime_value
 
    if request.method == "POST":
        uploaded_file = request.FILES.get('test_csv')  # 사용자가 업로드한 test_csv 파일 받아오기
        avg_val = request.POST.get('avg_value')  # 사용자가 입력한 평균값 받아오기
        real_values = request.POST.getlist('real')  # 'real'이라는 키로 전송된 여러 개의 데이터를 리스트로 받아옵니다.

        # 각 'real' 값을 WLifecycle 테이블에 저장
        for id_, real_value in zip(request.POST.getlist('ID'), request.POST.getlist('real')):
            instance = WLifecycle(test_csv=uploaded_file, Lot_ID=id_, avg_value=float(avg_val), real=float(real_value))
            instance.save()
        
        # POST 요청이 오는 경우에만 prediction_data를 다시 가져와야 함
        # prediction_data 초기화
        try:
            latest_prediction = Prediction_complete.objects.latest('ID')
            prediction_file = latest_prediction.csv_file.path
            prediction_data = pd.read_csv(prediction_file)
        except (Prediction_complete.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            # 오류 발생 시, 기본 데이터를 생성해 prediction_data를 초기화
            prediction_data = pd.DataFrame()
            prediction_data = prediction_data.append(data, ignore_index=True)
            latest_prediction = Prediction_complete()
       
        # 'updated_data'를 정의: 사용자가 입력한 'real' 데이터를 리스트 형태로 가져오기
        updated_data = [{"ID": int(id_), "real": float(value)} for id_, value in zip(request.POST.getlist('ID'), request.POST.getlist('real'))]

        # 'real' 컬럼 데이터 추가
        for item in updated_data:
            prediction_data.loc[prediction_data['ID'] == item['ID'], 'real'] = item['real']
        
        # 델타 값 계산
        prediction_data['delta'] = (prediction_data['prediction'] - prediction_data['real']).abs()
        
        # 델타 값의 평균 계산
        avg_delta = prediction_data['delta'].mean()
        latest_prediction.avg_delta = avg_delta
        prediction_data['avg_del'] = avg_delta
        
        # 데이터를 Prediction_complete 테이블에 다시 저장
        prediction_csv = StringIO()
        prediction_data.to_csv(prediction_csv, index=False)
        latest_prediction.csv_file = ContentFile(prediction_csv.getvalue().encode('utf-8'), name="updated_predictions.csv")
        latest_prediction.save()
        
        # WLifecycle 테이블에서 마지막 행의 데이터 가져오기
        latest_wlifecycle = WLifecycle.objects.latest('real_input_time')
        latest_real_input_time = latest_wlifecycle.real_input_time
        latest_id = latest_wlifecycle.Lot_ID
        predictions = prediction_data['prediction'].tolist()
        
    return render(request, "hynix/lifecycle.html", {
        "avg_delta": avg_delta,
        "Date": latest_real_input_time,
        "ID": latest_id,
        "Pred": predictions
    })