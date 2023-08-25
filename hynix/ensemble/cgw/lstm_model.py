### test 코드 train에 독립적이도록 수정할 것

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchmetrics import R2Score
import pandas as pd
import math

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
        
        
def val_model(model, criterion1, criterion2, val_dataloader, device):
    model.eval()
    loss_out = 0
    with torch.no_grad(): # 자동적으로 grad를 저장하므로 쓸데없는 메모리 낭비를 막을 수 있음
        for data in val_dataloader:
            x = data["x"].to(device)
            x = x.to(torch.float)
            y = data["y"].to(device)
            y = y.to(torch.float)
            out = model(x)
            out = out.squeeze(1)
            
            loss1 = criterion1(out, y)
            loss2 = criterion2(out, y)
            loss = loss1 + (1-loss2)
            
            loss_out += loss.item()
        return loss_out / len(val_dataloader)
    
    
def train_model(model, train_dataloader, val_dataloader, device,optimizer_name,epochs,scheduler_name):

    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if scheduler_name == "reduce":
        # if optimizer_name != "sgd" and scheduler == "reduce":
        #     raise Exception("only sgd optimizer can use this scheduler")
        # else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, 
                                                                 T_mult=1, eta_min=0.00001)
    
    criterion1 = nn.MSELoss()
    criterion2 = R2Score()
    epochs = epochs
    train_loss_all = []
    val_loss_all = []
    r2_all = []
    best_loss = float("inf")
    
    lrs=[]

    # early stopping
    # patience_limit = 10 # 몇 번의 epoch까지 지켜볼지를 결정
    # patience_check = 0

    for epoch in range(epochs):
        loss_ep = 0
        total_r2loss = 0
        model.train()
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="{}/{}".format(epoch+1, epochs)):
            x = data["x"].to(device)
            x = x.to(torch.float)
            y = data["y"].to(device)
            y = y.to(torch.float)
            out = model(x)
            out = out.squeeze(1)

            optimizer.zero_grad()
            loss1 = criterion1(out, y)
            loss2 = criterion2(out, y)
            loss = loss1 + (1-loss2)
            loss.backward()
            optimizer.step()
            
            loss_ep += loss.item()
            total_r2loss += loss2.item()

        train_loss = loss_ep/len(train_dataloader)
        val_loss = val_model(model, criterion1, criterion2, val_dataloader, device)
        r2_loss = total_r2loss / len(train_dataloader)
        
        if scheduler_name == "reduce":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        
        print("train_loss = {:.10f}, val_loss = {:.10f}".format(train_loss, val_loss))
        print("r2 score = {:.10f}".format(r2_loss))
        
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        r2_all.append(r2_loss)
        
        if best_loss > val_loss:
            print("**** save best model ****")
            best_loss = val_loss
            torch.save(model, "lstm_best_model_{}_{}.pt".format(optimizer_name,scheduler_name))
         
        # patience_check = 0
        # else: # loss가 개선되지 않은 경우
        #     patience_check += 1
        #     if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
        #         break
        
    return train_loss_all, val_loss_all, r2_all, lrs
    
def test_model(test_loader, device,optimizer_name,scheduler_name):
    pred = []
    model = load_best_model(optimizer_name,scheduler_name)
    with torch.no_grad():
        for data in test_loader:
            x = data["x"].to(device)
            x = x.to(torch.float)
                
            prediction = model(x)
            pred.append(prediction)
        return pred
        
def load_best_model(optimizer_name, scheduler_name):
    return torch.load("lstm_best_model_{}_{}.pt".format(optimizer_name,scheduler_name))

def predict(device, test_dataloader, optimizer_name, scheduler_name, save_path):
    model = load_best_model(optimizer_name, scheduler_name)
    
    outputs = []
    real = []
    for data in test_dataloader:
        x = data["x"].to(device)
        x = x.to(torch.float)
        
        y = data["y"].to(device)
        y = y.to(torch.float)

        output = model(x)
        output = torch.flatten(output, 0)

        outputs.append(output)
        real.append(y)

    outputs = torch.cat(outputs, dim=0) * 100
    real = torch.cat(real, dim=0) * 100
    
    mse_loss = nn.MSELoss()
    loss = mse_loss(outputs, real)
    r2_score = R2Score()
    r2 = r2_score(outputs, real)
    
    df = pd.DataFrame()
    df["Y"] = real.detach().numpy() 
    df["Prediction"] = outputs.detach().numpy() 
    
    lmplot("Prediction", "Y", df, optimizer_name,scheduler_name, save_path)
    
    return loss, r2