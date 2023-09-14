from .models import PreprocessedCSV, Prediction_complete, WLifecycle, Wsimulation
from django.db import models as dj_models
from io import StringIO
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
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
        isFull = 0
        if "isFull" in request.POST:
            isFull = int(request.POST["isFull"])
        input_csv_instance = Wsimulation(test_csv=test_data_file)
        input_csv_instance.save()
        
        uploaded_file_path = input_csv_instance.test_csv.path
        test_data = pd.read_csv(uploaded_file_path, index_col=0)
        print(test_data.columns)
        
        lot_id_value = test_data['ID'].iloc[0] if 'ID' in test_data.columns else None
        
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
        print(min_val, max_val, mean_val)
        print("confidence interval complete")
        
        # 결과를 리스트로 저장
        input_csv_instance.min_value = min_val
        input_csv_instance.max_value = max_val
        input_csv_instance.avg_value = mean_val
        input_csv_instance.last_filled_column_name = last_filled_column_name
        input_csv_instance.save()
        
        # 1번: test_data 파일 업로드 시 WLifecycle에 저장 (조건부)
        if isFull == 1 and lot_id_value != None:
            lifecycle_instance = WLifecycle(Lot_ID=lot_id_value, test_csv=test_data_file, avg_value=mean_val)
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



from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt # 보안 및 공격 방지
def lifecycle(request):
    context = {} # 템플릿에 전달될 데이터를 담기. 지금은 초기화.

    def get_data_from_database(): # DB로부터 데이터를 가져오는 역할
        w_lifecycle_data_list = WLifecycle.objects.all().values('Lot_ID', 'avg_value', 'real', 'real_input_time').order_by('real_input_time')
        
        # 데이터가 DB에 존재하는 경우
        if w_lifecycle_data_list.exists(): # DB에 WLifecycle 데이터가 있는지 확인
            return list(w_lifecycle_data_list)
        
        # 데이터가 DB에 존재하지 않는 경우
        else:
            data = []
            from datetime import datetime, timedelta
            start_date = datetime.strptime("2023-08-23 12", "%Y-%m-%d %H")
            for i in range(1,31):
                date_value = start_date + timedelta(days=i - 1)
                lot_id = 1+i
                pred = 88
                real = 88
                data.append({
                    "Lot_ID": lot_id, # "" 안에 글자가 화면에 표시되는 글자
                    "avg_value": pred, 
                    "real": real,
                    "real_input_time": date_value.strftime("%Y-%m-%d %H")
                })
            return data

    data = get_data_from_database()

    # 1. GET(표)
    if request.method == "GET":
        context = {"data": data}

    # 2. POST(그래프)
    elif request.method == "POST":
        try:
            updated_data_list = json.loads(request.POST['IDreal'])
        except json.JSONDecodeError as e:
            print("Failed to decode JSON data:", e)
        
        # POST로 받은 업데이트된 데이터로 WLifecycle DB를 업데이트
        for updated_item in updated_data_list:
            lot_id = updated_item["Lot_ID"]
            real_value = float(updated_item["real"])
            if "년" in updated_item["real_time"]:
                date_string = updated_item["real_time"]
                date_string = date_string.replace("년","").replace("일","")
                month_mapping = {
                    '1월': '01',
                    '2월': '02',
                    '3월': '03',
                    '4월': '04',
                    '5월': '05',
                    '6월': '06',
                    '7월': '07',
                    '8월': '08',
                    '9월': '09',
                    '10월': '10',
                    '11월': '11',
                    '12월': '12'
                }

                # 월과 시간대 문자열을 대응하는 영어로 변경
                for k, v in month_mapping.items():
                    date_string = date_string.replace(k, v)
                date_string = date_string.replace('오후', 'PM').replace('오전', 'AM')

                # datetime 객체로 변환
                date_format = '%Y %m %d %I:%M %p'
                date_object = datetime.strptime(date_string, date_format)

                # 원하는 포맷으로 출력
                desired_format = '%Y-%m-%d %H:%M:%S'
                real_input_time = date_object.strftime(desired_format)
                
                print("not none",real_input_time)
            else:
                real_input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("none",real_input_time)
            lot_id = int(lot_id)
            
            # WLifecycle에서 Lot_ID 기반으로 항목을 찾아 데이터 갱신
            try:
                w_lifecycle_entry = WLifecycle.objects.get(Lot_ID=lot_id)
                w_lifecycle_entry.real = real_value
                w_lifecycle_entry.real_input_time = real_input_time
                w_lifecycle_entry.save()
            except WLifecycle.DoesNotExist:
                print(f"No entry found for Lot_ID: {lot_id}")
         
        # POST 이후 데이터베이스의 최신 데이터를 가져옵니다.
        data = get_data_from_database()

        # 처음에 가져온 데이터를 기반으로 델타 값을 계산
        deltas = [
            abs((item["avg_value"] - item["real"])) if ("real" in item and "avg_value" in item and item["real"] is not None and item["avg_value"] is not None) else None
            for item in data
        ]
        deltas = json.dumps(deltas)
        try:
            real_input_time = [item.get("real_input_time", None).strftime('%Y-%m-%d %H') for item in data]
        except:
            real_input_time = [item.get("real_input_time", None) for item in data]
        real_input_time= json.dumps(real_input_time)
        
        data = {
            "avg_delta": deltas,
            "real_input_time": real_input_time
        }
        print(data) 
        return JsonResponse(data)
    return render(request, "hynix/lifecycle.html", context)