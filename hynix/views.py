from django.db import models as dj_models
from io import StringIO
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Prediction
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

def main(request):
    return render(request, 'hynix/main.html',{"contents":"<h1>main page</h1>"})

# test = pd.read_csv('C:/Users/dykim/OneDrive/바탕 화면/공부자료/22, 23 AI 공부/2023 T아카데미 ASAC/기업 프로젝트/4. web_django/hynix_fab_project/hynix/ensemble/dataset')

def get_predictions(test_data):
    test = test_data
    # test.drop(columns=['Date'], inplace = True)

    # 동연
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
        
    test1 = test
    with open('hynix/ensemble/kdy/preprocess_funcs.pkl', "rb") as file:
        pf = pickle.load(file)

    deleted_columns, null_columns, to_drop_all = pf.load_pickles()
    final_test = pf.preprocessing_realtest(test1)
    scaled_final_test = pf.scale_data_without_target(final_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('hynix/ensemble/kdy/LSTM.pkl', "rb") as file:
        LSTM = pickle.load(file)

    model = torch.load('hynix/ensemble/kdy/3. final_best_model.pt')
    model.eval()

    test_tensor = torch.tensor(scaled_final_test.values).float().to(device)
    test_tensor = test_tensor.view(-1, 1, 1285)

    with torch.no_grad():
        predictions = model(test_tensor)

    predictions_test = predictions.cpu().numpy()
    predictions_test = predictions_test * 100

    kdy_pred = pd.DataFrame(predictions_test) # 833 rows × 1 columns
    
    #-------------------------------------------------------------------
    # 고운

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
        def sort_time(self, data):
            for idx,col in enumerate(data.columns):
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
                df[ts_data.columns[idx]+"-"+ts_data.columns[idx-1]] = col
            return df

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

            pickle.dump(std, open('hynix/ensemble/cgw/std_scaler.pkl', 'wb'))
            
            corr_df = train.apply(lambda x: x.corr(y_train))
            corr_df = corr_df.apply(lambda x: round(x ,2))
            df = pd.DataFrame(corr_df[corr_df<1], columns=['corr'])
            cols = df[abs(df["corr"]) >= 0.05].index.to_list()
            train = train[cols]
            
            train_options["column_names"] = train.columns.to_list()
            train_options["column_means"] = list(train.mean().values)

            pickle.dump(train_options, open('hynix/ensemble/cgw/train_options.pkl', 'wb'))
            
            y_train /= 100
            train = pd.merge(train, y_train,how="left",on="ID") 

            return train

        def test_preprocess(self, test):
            print("test preprocess start")
            test.set_index(keys="ID",inplace=True)
            test.drop(columns="x204",inplace=True)
            
            test = self.sort_time(test)
            ts_test = test.select_dtypes("datetime").astype("str")
            test_q = self.Qtime(test, ts_test)
            test = self.insert_Qtime(test, test_q)
            
            train_options = pickle.load(open('hynix/ensemble/cgw/train_options.pkl', 'rb'))
            scaler = pickle.load(open('hynix/ensemble/cgw/std_scaler.pkl', 'rb'))
            
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
            test = self.RealTestDataset(test)
            test_loader = DataLoader(test, batch_size=833, shuffle=False, drop_last=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load('hynix/ensemble/cgw/lstm_best_model_sgd_cosine.pt')
            
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
        
    test2 = test.iloc[:, 1:]
    ppp = PreprocessAndPredict()
    cgw_pred = ppp.run(test2)
    cgw_pred = pd.DataFrame(cgw_pred.detach().numpy())
    #-------------------------------------------------------------------
    # 정우

    class Preprocessor :
        default_path = 'hynix/ensemble/ljw'

        def __init__(self):
            pass

        def time_preprocessing(self,df):
            df.set_index(keys="ID", inplace=True)
            df.drop(columns="x204",inplace=True)
            for idx,col in enumerate(df.columns.to_list()[188:1454]):
                if df[col].dtype == "object":
                    df[col] = pd.to_datetime(df[col])
            datatmp = df.columns.to_list()[188:1454]
            ts_data = df.select_dtypes("datetime")
            for idx in ts_data.index:
                ts_data.sort_values(by=idx,axis=1, inplace=True)

            columns = ts_data.columns
            num_columns = len(columns)
            for i in range(1, num_columns):
                column_diff = pd.to_datetime(df.iloc[:, i]) - pd.to_datetime(ts_data.iloc[:, i-1])
                column_diff_hours = column_diff.dt.total_seconds() / 3600
                ts_data.iloc[:, i] = column_diff_hours

            for i in range(len(ts_data.columns)-1):
                ts_data.iloc[:, i] = ts_data.iloc[:, i+1]
            result = []
            for idx,col in enumerate(datatmp):
                if df[col].dtype == "<M8[ns]":
                    cur = int(col[1:])
                    i = idx
                    tmp = []
                    while i > 0:
                        i -= 1
                        next = datatmp[i]
                        if df[next].dtype == "<M8[ns]":
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

            ts_final = df[ts_final]
            front = df.loc[:,:"x193"]
            back = df.loc[:,"x1461":]
            final = pd.concat([front, ts_final, back], axis = 1)
            ts_data.drop(columns=ts_data.columns[-1], inplace=True)

            for col in ts_data.columns:
                if col in final.columns:
                    final[col] = ts_data[col]
            final.drop(['x197'], axis=1, inplace=True)
            df = final
            return df

        def na_ratio_preprocessing_train(self,train):
            na_percentage = train.isna().sum() / len(train)
            high_na_columns = na_percentage[na_percentage >= 0.9].index
            train=train.drop(high_na_columns, axis=1,inplace=False)
            with open(Preprocessor.default_path + '/high_na_columns.pkl', 'wb') as f:
                pickle.dump(high_na_columns, f)
            return train

        def na_ratio_preprocessing_Rtest(self,Rtest):
            with open(Preprocessor.default_path + '/high_na_columns.pkl', 'rb') as f:
                high_na_columns = pickle.load(f)
            Rtest=Rtest.drop(high_na_columns, axis=1,inplace=False)
            return Rtest

        def correlation_preprocessing_train(self,train):
            numeric_columns = train.select_dtypes(include=['int64', 'float64'])
            correlation_with_target = numeric_columns.corr()['Y']
            columns_with_low_correlation = correlation_with_target[correlation_with_target.abs() < 0.04].index
            train.drop(columns_with_low_correlation, axis=1, inplace=True)
            with open(Preprocessor.default_path+'/columns_with_low_correlation.pkl', 'wb') as f:
                pickle.dump(columns_with_low_correlation, f)
            return train

        def correlation_preprocessing_Rtest(self,Rtest):
            with open(Preprocessor.default_path+'/columns_with_low_correlation.pkl', 'rb') as f:
                columns_with_low_correlation = pickle.load(f)
            Rtest.drop(columns_with_low_correlation, axis=1, inplace=True)
            return Rtest

        def replace_outliers_median_preprocessing_train(self,train , threshold=1.5):
            all_lower_bound={}
            all_upper_bound={}
            all_col_median={}
            for col in train.columns:
                Q1 = train[col].quantile(0.25)
                Q3 = train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                all_lower_bound[col]=lower_bound
                upper_bound = Q3 + threshold * IQR
                all_upper_bound[col]=upper_bound

                train_outliers = train[col][(train[col] < lower_bound) | (train[col] > upper_bound)]
                col_median = train[col].median()
                all_col_median[col]=col_median
                train.loc[train_outliers.index, col] = col_median
            with open(Preprocessor.default_path+'/all_lower_bound.pkl', 'wb') as f:
                pickle.dump(all_lower_bound, f)
            with open(Preprocessor.default_path+'/all_upper_bound.pkl', 'wb') as f:
                pickle.dump(all_upper_bound, f)
            with open(Preprocessor.default_path+'/all_col_median.pkl', 'wb') as f:
                pickle.dump(all_col_median, f)
            return train

        def replace_outliers_median_preprocessing_Rtest(self,Rtest, threshold=1.5):
            with open(Preprocessor.default_path+'/all_lower_bound.pkl', 'rb') as f:
                all_lower_bound = pickle.load(f)
            with open(Preprocessor.default_path+'/all_upper_bound.pkl', 'rb') as f:
                all_upper_bound = pickle.load(f)
            with open(Preprocessor.default_path+'/all_col_median.pkl', 'rb') as f:
                all_col_median = pickle.load(f)

            for col in Rtest.columns:
                Rtest_outliers = Rtest[col][(Rtest[col] < all_lower_bound[col]) | (Rtest[col] > all_upper_bound[col])]
                Rtest.loc[Rtest_outliers.index, col] = all_col_median[col]
            return Rtest

        def fillna_mean_preprocessing_train(self,train):
            col_means = {}
            use_col = []

            for col in train.columns:
                col_mean = train[col][~train[col].isnull()].mean()
                if pd.notna(col_mean):
                    train[col].fillna(col_mean, inplace=True)
                    use_col.append(col)
                    if train[col].nunique() == 1:
                        use_col.remove(col)
                        col_means[col] = col_mean
            with open(Preprocessor.default_path+'/col_means.pkl', 'wb') as f:
                pickle.dump(col_means, f)
            with open(Preprocessor.default_path+'/use_col.pkl', 'wb') as f:
                pickle.dump(use_col, f)
            return train[use_col]

        def fillna_mean_preprocessing_Rtest(self,Rtest):
            with open(Preprocessor.default_path+'/col_means.pkl', 'rb') as f:
                col_means = pickle.load(f)
            with open(Preprocessor.default_path+'/use_col.pkl', 'rb') as f:
                use_col = pickle.load(f)

            for col in Rtest.columns:
                if col in use_col:
                    Rtest[col].fillna(col_means[col], inplace=True)
            return Rtest[use_col]

        def scale_preprocessing_train(self,train):
            scaler = MinMaxScaler()
            strain = scaler.fit_transform(train)
            with open(Preprocessor.default_path+'/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            train = pd.DataFrame(strain, columns=train.columns,index=train.index)
            return train

        def scale_preprocessing_Rtest(self,Rtest):
            with open(Preprocessor.default_path+'/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            sRtest = scaler.transform(Rtest)
            Rtest = pd.DataFrame(sRtest, columns=Rtest.columns,index=Rtest.index)
            return Rtest

        def preprocessing_train(self, train):
            train = self.na_ratio_preprocessing_train(train)
            train = self.correlation_preprocessing_train(train)
            train = train.drop(columns=['Y'])
            train = self.replace_outliers_median_preprocessing_train(train)
            train = self.fillna_mean_preprocessing_train(train)
            train = self.scale_preprocessing_train(train)
            return train

        def preprocessing_Rtest(self,df):
            Rtest = self.time_preprocessing(df)
            Rtest = self.na_ratio_preprocessing_Rtest(Rtest)
            Rtest = self.correlation_preprocessing_Rtest(Rtest)
            Rtest = self.replace_outliers_median_preprocessing_Rtest(Rtest)
            Rtest = self.fillna_mean_preprocessing_Rtest(Rtest)
            Rtest = self.scale_preprocessing_Rtest(Rtest)
            return Rtest
        
    default_path = 'hynix/ensemble/ljw/'
    with open(default_path + "Preprocessor", "rb") as f:
        preprocess = pickle.load(f)
            
    test3 = test.iloc[:, 1:]
    Rtest = preprocess.preprocessing_Rtest(test3)

    best_model = load_model('hynix/ensemble/ljw/ML_best_model')
    ljw_pred = predict_model(best_model, data=Rtest)['prediction_label']*100
    ljw_pred = pd.DataFrame(ljw_pred)
        
    # ----------------------------------------------------------------------------
    # ensemble
    
    mse_values = [10.74511, 11.8428, 1.9966]
    weights = [1/mse for mse in mse_values]
    normalized_weights = [weight/sum(weights) for weight in weights]
    
    kdy_pred = kdy_pred.reset_index(drop=True)
    cgw_pred = cgw_pred.reset_index(drop=True)
    ljw_pred = ljw_pred.reset_index(drop=True)

    ensemble_module = kdy_pred.iloc[:, 0]*normalized_weights[0] + cgw_pred.iloc[:, 0]*normalized_weights[1] + ljw_pred.iloc[:, 0]*normalized_weights[2]
    predictions = ensemble_module
    predictions.columns = ['prediction']
    prediction = pd.concat([test_data, predictions], axis=1)
    return prediction

def process_file(test_data_file):
    test_data = pd.read_csv(test_data_file)
    predictions = get_predictions(test_data)
    return predictions

from sklearn.utils import resample

def calculate_confidence_interval(predictions, alpha=0.9):
    # Bootstrap re-sampling을 사용하여 신뢰구간 계산
    bootstrapped_samples = [resample(predictions) for _ in range(1000)]
    min = np.percentile(bootstrapped_samples, (1-alpha)/2*100)
    max = np.percentile(bootstrapped_samples, alpha+((1-alpha)/2)*100)
    return min, max

def simulation(request):
    test_data_html = "<h1>simulator page</h1>"
    prediction = None
    confidence_interval = {"min": None, "max": None}
    test_data_json = '{}'  

    if request.method == "POST":
        if 'test_data' in request.FILES:
            test_data_file = request.FILES['test_data']
            
            test_data = pd.read_csv(test_data_file)
            test_data_html = test_data.head(10).to_html()
            
            try:
                # 'unnamed 0' 컬럼 제거
                test_data = test_data.drop(columns=['Unnamed: 0'], errors='ignore')
                test_data.to_csv('hynix/read_csv/test_data.csv')
                
                # ID 컬럼을 key로 하는 json 생성
                test_data_json = test_data.head(10).set_index('ID').to_json(orient='index')
            except Exception as e:
                test_data_json = '{}'
                print(f"Error converting data to JSON: {e}")
            
        if 'run' in request.POST:
            run_timestamp = request.POST['run_timestamp']  # Run 버튼 누른 시점의 시간 값을 받아옴
            test_data_file = pd.read_csv('hynix/read_csv/test_data.csv')
            prediction = process_file(test_data_file)
            
            # 신뢰구간 계산
            min, max = calculate_confidence_interval(prediction["prediction"])
            confidence_interval = {"min": min, "max": max}
            
            # 'Date' 컬럼 추가하고 Run 버튼 누른 시점의 시간 값으로 채움
            prediction['Date'] = run_timestamp
            
            if 'yes' in request.POST:
                prediction_csv = StringIO()
                prediction.to_csv(prediction_csv, index=False)
                prediction_model = Prediction(csv_file=ContentFile(prediction_csv.getvalue().encode('utf-8'), name="predictions.csv"))
                prediction_model.save()
            
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
            return response

    return render(request, "hynix/simulation.html", {"contents": test_data_html, "prediction": prediction, "data_json": test_data_json, "confidence_interval": confidence_interval})





from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_protect
@csrf_protect

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
            latest_prediction = Prediction.objects.latest('date_created')
            prediction_file = latest_prediction.csv_file.path
            prediction_data = pd.read_csv(prediction_file)
        except (Prediction.DoesNotExist, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
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
        Date = prediction_data['Date'] 
        
        # 데이터를 DB에 다시 저장
        
        # 데이터와 함께 lifecycle.html 페이지를 렌더링해 화면에 보여줌
        return render(request, "hynix/lifecycle.html", {"avg_delta": avg_delta, "Date": Date.to_string() })


        # 데이터와 함께 lifecycle.html 페이지를 렌더링해 화면에 보여줌
    return render(request, "hynix/lifecycle.html", {"data": data})