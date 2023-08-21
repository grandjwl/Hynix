import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, TensorDataset

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1234)

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
        with open('C:/Users/dykim/OneDrive/바탕 화면/공부자료/22, 23 AI 공부/2023 T아카데미 ASAC/기업 프로젝트/3. 최종 모델링 파일/pickle_save/train_q_train.pkl', 'wb') as f:
            pickle.dump(df, f)
        return df

    def delete_null1(self, final):
        empty_columns = final.columns[final.isnull().all()]
        final = final.drop(empty_columns, axis=1)
        self.deleted_columns = list(empty_columns)
        with open('C:/Users/dykim/OneDrive/바탕 화면/공부자료/22, 23 AI 공부/2023 T아카데미 ASAC/기업 프로젝트/3. 최종 모델링 파일/pickle_save/deleted_columns.pkl', 'wb') as f:
            pickle.dump(self.deleted_columns, f)
        return final, self.deleted_columns

    def delete_null2(self, final):
        null_threshold = 0.95 
        null_counts = final.isnull().sum() 
        total_rows = final.shape[0]
        self.null_columns = null_counts[null_counts / total_rows >= null_threshold].index.tolist()
        final = final.drop(self.null_columns, axis=1)
        with open('C:/Users/dykim/OneDrive/바탕 화면/공부자료/22, 23 AI 공부/2023 T아카데미 ASAC/기업 프로젝트/3. 최종 모델링 파일/pickle_save/null_columns.pkl', 'wb') as f:
            pickle.dump(self.null_columns, f)    
        return final, self.null_columns
    
    def fillna_null(self, final):
        final.drop(final.columns[0], axis=1, inplace = True)
        final.drop('ID', axis = 1, inplace = True)
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
        with open('C:/Users/dykim/OneDrive/바탕 화면/공부자료/22, 23 AI 공부/2023 T아카데미 ASAC/기업 프로젝트/3. 최종 모델링 파일/pickle_save/to_drop_all.pkl', 'wb') as f:
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
        with open("./pickle_save/deleted_columns.pkl", "rb") as file:
            deleted_columns = pickle.load(file)
        with open("./pickle_save/null_columns.pkl", "rb") as file:
            null_columns = pickle.load(file)
        with open("./pickle_save/to_drop_all.pkl", "rb") as file:
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
    
test = pd.read_csv(r"C:\Users\dykim\OneDrive\바탕 화면\공부자료\22, 23 AI 공부\2023 T아카데미 ASAC\기업 프로젝트\1. machine learning\dataset\testset.csv")

with open("./pickle_save/preprocess_funcs.pkl", "rb") as file:
    pf = pickle.load(file)

deleted_columns, null_columns, to_drop_all = pf.load_pickles()
final_test = pf.preprocessing_realtest(test)
scaled_final_test = pf.scale_data_without_target(final_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("./pickle_save/LSTM.pkl", "rb") as file:
    LSTM = pickle.load(file)

model = torch.load('./model_save/3. final_best_model.pt')
model.eval()

test_tensor = torch.tensor(scaled_final_test.values).float().to(device)
test_tensor = test_tensor.view(-1, 1, 1285)

with torch.no_grad():
    predictions = model(test_tensor)

predictions_test = predictions.cpu().numpy()
predictions_test = predictions_test * 100

predictions_df = pd.DataFrame(predictions_test)
predictions_df.to_csv('C:/Users/dykim/OneDrive/바탕 화면/공부자료/22, 23 AI 공부/2023 T아카데미 ASAC/기업 프로젝트/prediction/4.FINAL_PREDICT.csv', index=False)