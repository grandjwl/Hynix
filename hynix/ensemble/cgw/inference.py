import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchmetrics import R2Score
import torch
from torch.utils.data import DataLoader
import pickle
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from pycaret.regression import * 

# random seed 고정
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1234)
import warnings
warnings.filterwarnings('ignore')

class PreprocessAndPredict():
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
        
    def __init__(self):
        pass
    # 시간 순 정렬
    def sort_time(self, data):
        # 계측 시간 데이터 object -> datetime
        for idx,col in enumerate(data.columns):
            if data[col].dtype == "object":
                data[col] = pd.to_datetime(data[col])

        ts_data = data.select_dtypes("datetime")

        # 날짜 오름차순 정렬
        for idx in ts_data.index:
            ts_data.sort_values(by=idx,axis=1, inplace=True)

        # (시간, 계측값) 정렬
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

    # qtime 계산 
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
            df[ts_data.columns[idx]+"-"+ts_data.columns[idx-1]] = col
        return df

    # qtime 끼워넣기
    def insert_Qtime(self, data, data_q):
        idx = 0
        for col in data.columns:
            try:
                if data.loc[:,col].dtype == "datetime64[ns]":
                    data.loc[:,col] = data_q.iloc[:,idx].values
                    idx += 1
            except:
                break
        data.drop(columns="x197",inplace=True)
        
        return data

    # train 전처리
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
        
        # 결측치 처리
        head = train.loc[:,:"x193"]
        mid = train.loc[:,"x205":"x196"]
        tail = train["x1548"]
        
        for elem in head.columns:
        # sensor 값의 not null값의 개수가 5보다 작은 컬럼 제거
            if head[elem].notnull().sum() < 5:
                head.drop(columns=elem,inplace=True)

        # 정규성 확인 (shapiro-wilk)
        sw = []
        sw_pvalues = []

        for col in head.columns:
            x = head[head[col].notnull()][col]

            test_stat, p_val = stats.shapiro(x)
            sw.append((col,p_val))
            sw_pvalues.append(p_val)

        # 정규성 불만족 컬럼명 추출
        no_cols = []
        for col,val in sw:
            if val < 0.05:
                no_cols.append(col)

        # 정규성 만족 컬럼명 추출
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

        # nan 값의 비율이 크지 않아 평균으로 대치
        for col in mid.columns:
            mid[col].fillna(mid[col].mean(), inplace=True)

        train = pd.concat([head,mid,tail],axis=1)
        train_options["before_scale_columns"] = train.columns.to_list()
        
        # 5. 스케일링
        std = StandardScaler()
        std.fit(train)

        train_sc = std.transform(train)
        train = pd.DataFrame(data=train_sc, index=train.index, columns=train.columns)

        pickle.dump(std, open('./std_scaler.pkl', 'wb'))
        
        # train 상관계수
        corr_df = train.apply(lambda x: x.corr(y_train))
        corr_df = corr_df.apply(lambda x: round(x ,2))
        df = pd.DataFrame(corr_df[corr_df<1], columns=['corr'])
        cols = df[abs(df["corr"]) >= 0.05].index.to_list()
        train = train[cols]
        
        train_options["column_names"] = train.columns.to_list()
        train_options["column_means"] = list(train.mean().values)

        pickle.dump(train_options, open('./train_options.pkl', 'wb'))
        
        y_train /= 100
        train = pd.merge(train, y_train,how="left",on="ID") 

        return train

    # test 전처리
    def test_preprocess(self, test):
        print("test preprocess start")
        test.set_index(keys="ID",inplace=True)
        test.drop(columns="x204",inplace=True)
        
        test = self.sort_time(test)
        ts_test = test.select_dtypes("datetime").astype("str")
        test_q = self.Qtime(test, ts_test)
        test = self.insert_Qtime(test, test_q)
        
        train_options = pickle.load(open('./train_options.pkl', 'rb'))
        scaler = pickle.load(open('./std_scaler.pkl', 'rb'))
        
        # 스케일링 전 컬럼들 가져오기
        test = test[train_options["before_scale_columns"]]
        
        # 스케일링 trasform
        test_sc = scaler.transform(test)
        test = pd.DataFrame(data=test_sc, index=test.index, columns=test.columns)
        
        # 상관계수 이후 컬럼들 가져오기
        test = test[train_options["column_names"]]
        
        # 결측치 대치
        mean_values = train_options["column_means"]
        for idx, col in enumerate(test.columns.to_list()):
            test[col].fillna(mean_values[idx], inplace=True)
        
        # 최종 컬럼
        test = test[train_options["column_names"]]
        
        return test

    # 모델 예측값 생성 / 재학습
    def run(self, data):
        print("start test")
        test = self.test_preprocess(data)
        test = self.RealTestDataset(test)
        test_loader = DataLoader(test, batch_size=833, shuffle=False, drop_last=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.load('lstm_best_model_sgd_cosine.pt')
        
        outputs = []
        real = []
        for data in test_loader:
            x = data["x"].to(device)
            x = x.to(torch.float)

            output = model(x)
            output = torch.flatten(output, 0)

            outputs.append(output)

        outputs = torch.cat(outputs, dim=0) * 100
        return outputs