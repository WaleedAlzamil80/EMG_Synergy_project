import torch
from torch import nn

class WeightNormConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, use_bias, padding):
        super(WeightNormConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=use_bias)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.batch_norm_1 = nn.BatchNorm1d(12)
        self.dense_1 = nn.Linear(12, 1728) 
        self.reshape = lambda x: x.view(-1, 12, 144)
        self.dropout_2 = nn.Dropout(0.1)
        self.conv1d_1 = WeightNormConv1D(12, 8, kernel_size=3, activation=nn.SiLU(), use_bias=False, padding='same')
        self.avg_pooling = nn.AvgPool1d(kernel_size=2) 
        self.dropout_3 = nn.Dropout(0.1)
        self.conv1d_2 = WeightNormConv1D(8, 4, kernel_size=3, activation=nn.SiLU(), use_bias=True, padding='same')
        self.dropout_4 = nn.Dropout(0.1)
        self.conv1d_3 = WeightNormConv1D(4, 2, kernel_size=3, activation=nn.SiLU(), use_bias=True, padding='same')
        self.max_pooling = nn.MaxPool1d(kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.dense_2 = nn.Linear(70, 14)

    def forward(self, x):
        x = self.batch_norm_1(x)
        x = self.dense_1(x)
        x = self.reshape(x)
        x = self.dropout_2(x)
        x = self.conv1d_1(torch.unsqueeze(x, dim=1) )
        x = self.avg_pooling(x)
        x = self.dropout_3(x)
        x = self.conv1d_2(x)
        x = self.dropout_4(x)
        x = self.conv1d_3(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dense_2(x)
        return x
