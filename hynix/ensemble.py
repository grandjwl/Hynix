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

import warnings
warnings.filterwarnings(action='ignore')


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 동연


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
        
    def __init__(self):
        pass
    
    def sort_time(self, data):
        for idx,col in enumerate(data.columns):
            if data[col].dtype == "object":
                data[col] = pd.to_datetime(data[col])

        ts_data = data.select_dtypes("datetime")
        org_idx = ts_data.index.unique().to_list()[0]
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
        # drop_col = data.columns[188:1454].to_list()[-1]
        # data.drop(columns=drop_col,inplace=True)
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
        model = torch.load('hynix\ensemble\cgw\gw_final_lstm.pt')
        
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
class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_len = 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def reset_hid_cell(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.sequence_len, self.hidden_size),
            torch.zeros(self.num_layers, self.sequence_len, self.hidden_size))
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        # output = self.fc(output[:, -1, :])
        return output
        # return output.view(-1, self.output_size)

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
    
