from .models import PreprocessedCSV, Prediction_complete, WLifecycle, Wsimulation
from django.db import models as dj_models
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch.backends.cudnn as cudnn
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pycaret.regression import *
from sklearn.utils import resample
import warnings
warnings.filterwarnings(action='ignore')


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1234)

# 고운
class PreprocessAndPredict:
    class RealTestDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            
        def __len__(self):
            return len(self.df)
            
        def __getitem__(self,index):
            x = self.df.iloc[index, :].values
            data = {}
            data["x"] = x
            return data
        
    def __init__(self, isfull:bool):
        self.isfull = isfull
        self.nan = np.NaN
        
    # 50번의 시뮬레이션 데이터 생성
    def MakeSimulationData(self, test):
        csv_obj = PreprocessedCSV.objects.filter(data__contains='prepro_cgw.csv').first()
        if csv_obj:
            file_path = csv_obj.data.path
        # 파일을 직접 Pandas 데이터프레임으로 읽기
        org_traindata_df = pd.read_csv(file_path,index_col=0)
        org_traindata_df.drop(columns="Y",inplace=True)

        # test2의 행을 50번 복제
        test = pd.concat([test]*100).reset_index(drop=True)

        # 각 행에 대해 처리
        for i in range(len(test)):
            # 해당 행에서 마지막으로 값이 있는 컬럼 찾기
            last_valid_col = test.iloc[i].last_valid_index()

            # last_valid_col 다음 컬럼부터 값을 채우기
            for col in test.columns[test.columns.get_loc(last_valid_col)+1:]:
                try:
                    non_null_values = org_traindata_df[col].dropna().tolist()
                    if non_null_values:
                        random_value = random.choice(org_traindata_df[col].tolist())
                        test[col].iloc[i] = random_value
                except:
                    continue
        return test
    
    def sort_time(self, data):
        for idx,col in enumerate(data.columns):
            if data[col].dtype == "object":
                data[col] = pd.to_datetime(data[col])

        ts_data = data.select_dtypes("datetime")
        ts_data.reset_index(drop=True, inplace=True)

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
        final = pd.concat([front, ts_final, back], axis = 1)
        
        return final

    def Qtime(self, data, ts_data):
        df = pd.DataFrame(index=data.index)
        for idx in range(1, len(ts_data.columns)):
            col = []
            for jdx in range(len(ts_data.index)):
                try:
                    time1 = datetime.strptime(ts_data.iloc[jdx,idx],"%Y-%m-%d %H:%M")
                    time2 = datetime.strptime(ts_data.iloc[jdx,idx-1],"%Y-%m-%d %H:%M")
                except:
                    time1 = datetime.strptime(ts_data.iloc[jdx,idx],"%Y-%m-%d %H:%M:%S")
                    time2 = datetime.strptime(ts_data.iloc[jdx,idx-1],"%Y-%m-%d %H:%M:%S")

                diff =  time1 - time2
                col.append(round(diff.seconds/(60*60),2))
            df[ts_data.columns[idx-1]] = col
        return df

    def insert_Qtime(self, data, data_q):
        for col in data_q.columns:
            data.loc[:,col] = data_q.loc[:,col]
        if self.isfull:
            data.drop(columns="x197",inplace=True)
        else:
            try:
                last = data.select_dtypes("object").columns[-1]
                data[last] = self.nan
            except:
                pass
        
        return data

    def train_preprocess(self, train):
        print("train preprocess start")
        train.set_index(keys="ID",inplace=True)
        train.drop(columns="x204",inplace=True)
        y_train = train["Y"]
        train = train.drop(columns="Y")
        
        train_options = {}
        
        train = self.sort_time(train)
        ts_train = train.select_dtypes("datetime").astype("str")
        train_q = self.Qtime(train, ts_train)
        train = self.insert_Qtime(train, train_q)
        
        head = train.loc[:,:"x193"]
        mid = train.loc[:,"x205":"x196"]
        tail = train["x1548"]
        
        for elem in head.columns:
            if head[elem].notnull().sum() < 5:
                head.drop(columns=elem,inplace=True)

        sw = []
        sw_pvalues = []

        for col in head.columns:
            x = head[head[col].notnull()][col]

            test_stat, p_val = stats.shapiro(x)
            sw.append((col,p_val))
            sw_pvalues.append(p_val)

        no_cols = []
        for col,val in sw:
            if val < 0.05:
                no_cols.append(col)

        y_cols = []
        for col,val in sw:
            if val >= 0.05:
                y_cols.append(col)
        head = head[y_cols]

        nulldf = head.isnull().copy()
        for col in head.columns:
            for row in head.index:
                if nulldf.loc[row,col] == True:
                    head.loc[row,col] = head[col].mean()+np.random.randn()

        for col in mid.columns:
            mid[col].fillna(mid[col].mean(), inplace=True)

        train = pd.concat([head,mid,tail],axis=1)
        train_options["before_scale_columns"] = train.columns.to_list()
        
        std = StandardScaler()
        std.fit(train)

        train_sc = std.transform(train)
        train = pd.DataFrame(data=train_sc, index=train.index, columns=train.columns)

        pickle.dump(std, open('std_scaler.pkl', 'wb'))
        
        corr_df = train.apply(lambda x: x.corr(y_train))
        corr_df = corr_df.apply(lambda x: round(x ,2))
        df = pd.DataFrame(corr_df[corr_df<1], columns=['corr'])
        cols = df[abs(df["corr"]) >= 0.05].index.to_list()
        train = train[cols]
        
        train_options["column_names"] = train.columns.to_list()
        train_options["column_means"] = list(train.mean().values)

        pickle.dump(train_options, open('train_options.pkl', 'wb'))
        
        y_train /= 100
        train = pd.merge(train, y_train,how="left",on="ID") 

        return train

    def test_preprocess(self, test):
        print("test preprocess start")
        test.set_index(keys="ID",inplace=True)
        test.drop(columns="x204",inplace=True)
        
        if self.isfull:
            print("full data")
            test = self.sort_time(test)
            ts_test = test.select_dtypes("datetime").astype("str")
            test_q = self.Qtime(test, ts_test)
            test = self.insert_Qtime(test, test_q)
        else:
            print("middle data")
            cols = pickle.load(open('models/gw/train_cols.pkl', 'rb'))
            test = test[cols]
            ts_test = test.select_dtypes("object").astype("str")
            test_q = self.Qtime(test, ts_test)
            test = self.insert_Qtime(test, test_q)
        
        test = self.MakeSimulationData(test)
        
        train_options = pickle.load(open('models/gw/train_options.pkl', 'rb'))
        scaler = pickle.load(open('models/gw/std_scaler.pkl', 'rb'))
        
        test = test[train_options["before_scale_columns"]]
        
        test_sc = scaler.transform(test)
        test = pd.DataFrame(data=test_sc, index=test.index, columns=test.columns)
        
        test = test[train_options["column_names"]]
        
        mean_values = train_options["column_means"]
        for idx, col in enumerate(test.columns.to_list()):
            test[col].fillna(mean_values[idx], inplace=True)
        
        test = test[train_options["column_names"]]
        
        return test

    def run(self, data):
        print("start test")
        test = self.test_preprocess(data)
        model = load_model('models/gw/voting')
        print("load model complete")
        outputs = model.predict(test) * 100
        return outputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 고운
def model2_prediction(test_data, isFull):
    pp = PreprocessAndPredict(isfull=isFull)
    pred = pp.run(test_data)
    pred = pd.DataFrame(pred)
    return pred

def ensemble_models(test_data, isFull):
    predictions = model2_prediction(test_data, isFull) #고운
    predictions = predictions.reset_index(drop=True)

    return predictions

def calculate_confidence_interval(predictions, alpha=0.9):
    # 예측 값들의 평균 및 표준편차 계산
    mean_val = predictions.mean()
    std_val = predictions.std()
    var_val = predictions.var()

    # 90% 신뢰구간의 z-점수 계산 (정규 분포)
    z_score = stats.norm.ppf(1 - (1 - alpha) / 2)
    
    # 표준 오차 계산
    standard_error = std_val / np.sqrt(len(predictions))
    
    # 신뢰구간 계산
    margin_of_error = z_score * standard_error
    min_val = mean_val - margin_of_error
    max_val = mean_val + margin_of_error

    return min_val, max_val, mean_val