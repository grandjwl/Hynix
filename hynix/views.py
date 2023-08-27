from .models import PreprocessedCSV, Prediction_complete, WLifecycle, Wsimulation
from django.db import models as dj_models
from io import StringIO
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from hynix.model_class import LSTM, LSTM_model
from hynix.ensemble import ensemble_models, calculate_confidence_interval

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
        isFull = int(request.POST["isFull"])

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