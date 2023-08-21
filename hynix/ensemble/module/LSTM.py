import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        
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