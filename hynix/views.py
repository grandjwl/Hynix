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
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from datetime import datetime, timedelta

def main(request):
    return render(request, 'hynix/main.html',{"contents":""})

def simulation(request):
    prediction = None
    confidence_interval = {"min": [], "max": [], "avg": [], "process": []}
    test_data_json = '{}'  
    test_data = None
    last_filled_column_name = None
    all_last_filled_columns = None
    
    if request.method == "POST" and 'test_data' in request.FILES:
        test_data_file = request.FILES['test_data']
        isFull = int(request.POST["isFull"])

        input_csv_instance = Wsimulation(test_csv=test_data_file)
        input_csv_instance.save()
        
        uploaded_file_path = input_csv_instance.test_csv.path
        test_data = pd.read_csv(uploaded_file_path, index_col=0)
        
        lot_id_value = test_data['ID'].iloc[0] if 'ID' in test_data.columns else None
        
        # is_full_data 값이 True인 경우 WLifecycle 테이블에도 저장
        if isFull == 1 and lot_id_value != None:
            lifecycle_instance = WLifecycle(Lot_ID=lot_id_value, test_csv=test_data_file)
            lifecycle_instance.save()
        
        # 첫 번째 행에서 마지막으로 값이 있는 컬럼의 이름을 찾기 (멘토님이랑 상의 : 변경 예정)
        first_row = test_data.iloc[0].dropna()
        if not first_row.empty:
            last_filled_column_name = first_row.index[-1]
        # test_data를 JSON 형태로 변환
        test_data_temp = test_data
        test_data_temp = test_data_temp.drop(columns=['Unnamed: 0'], errors='ignore')
        test_data_json = test_data_temp.to_json(orient='records')
        
        prediction_df = ensemble_models(test_data, isFull)
        
        prediction = prediction_df.values
        print("ensemble complete")
        
        # 신뢰구간 계산
        min_val, max_val, mean_val = calculate_confidence_interval(prediction)
        print("confidence interval complete")
        
        # 결과를 리스트로 저장
        simulation_instance = Wsimulation(min_value=min_val, max_value=max_val, avg_value=mean_val, last_filled_column_name=last_filled_column_name)
        lifecycle_instance = WLifecycle(avg_value=mean_val)
        simulation_instance.save()
        lifecycle_instance.save()

        prediction_csv = StringIO()
        prediction_df.to_csv(prediction_csv, index=False)

        # 예측 결과를 Prediction_complete 모델에만 저장
        prediction_complete_model = Prediction_complete(name="Prediction Result", csv_file=ContentFile(prediction_csv.getvalue().encode('utf-8'), name="complete_predictions.csv"))
        prediction_complete_model.save()
        
    # Wsimulation 테이블에서 모든 [신뢰구간] 값 가져옴
    all_simulations = Wsimulation.objects.all()
    
    for sim in all_simulations: # [신뢰구간] 값을 리스트로 추가
        confidence_interval["min"].append(sim.min_value)
        confidence_interval["max"].append(sim.max_value)
        confidence_interval["avg"].append(sim.avg_value)
        confidence_interval["process"].append(sim.last_filled_column_name)
    confidence_interval = json.dumps(confidence_interval)
    return render(request, "hynix/simulation.html", {"prediction": prediction, "data_json": test_data_json, "confidence_interval": confidence_interval})

@csrf_exempt
def lifecycle(request):
    context = {}

    def get_data_from_database():
        try:
            w_lifecycle_data_list = WLifecycle.objects.all().values('Lot_ID', 'avg_value', 'real', 'real_input_time')
            data = list(w_lifecycle_data_list)
        except (WLifecycle.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            data = []
            start_date = datetime.strptime("2023-08-23", "%Y-%m-%d")
            for i in range(1, 101):
                date_value = start_date + timedelta(days=i - 1)
                lot_id = 1000 + i
                pred = 70 + i
                real = 70
                data.append({"Lot_ID": lot_id, "avg_value": pred, "real": real,"real_input_time": date_value.strftime("%Y-%m-%d")})
        return data

    data = get_data_from_database()

    # 1. GET(표)
    if request.method == "GET":
        print('GET')
        context = {"data": data}

    # 2. POST(그래프)
    elif request.method == "POST":
        print("POST")
        try:
            updated_data_list = json.loads(request.POST['IDreal'])
        except json.JSONDecodeError as e:
            print("Failed to decode JSON data:", e)

        # POST로 받은 업데이트된 데이터로 WLifecycle DB를 업데이트
        for updated_item in updated_data_list:
            lot_id = updated_item["Lot_ID"]
            real_value = int(updated_item["real"])
            lot_id = int(lot_id)
            real_value = int(real_value)
            real_input_time = datetime.now().strftime("%Y-%m-%d")
            
            # WLifecycle에서 Lot_ID 기반으로 항목을 찾아 데이터 갱신
            try:
                w_lifecycle_entry = WLifecycle.objects.get(Lot_ID=lot_id)
                w_lifecycle_entry.real = real_value
                w_lifecycle_entry.real_input_time = real_input_time
                w_lifecycle_entry.save()
            except WLifecycle.DoesNotExist:
                print(f"No entry found for Lot_ID: {lot_id}")

        # 처음에 가져온 데이터를 기반으로 델타 값을 계산
        deltas = [(item["avg_value"] - item["real"]) if item["real"] is not None else None for item in data]
        real_input_time = [item["real_input_time"] for item in data]
        
        data = {
            "avg_delta": deltas,
            "real_input_time": real_input_time
        }
        return JsonResponse(data)
    
    print(request.method)
    return render(request, "hynix/lifecycle.html", context)