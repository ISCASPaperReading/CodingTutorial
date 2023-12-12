import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self,seq_len,pred_len):
        super(TestNet, self).__init__()
        self.L = nn.Linear(seq_len,pred_len,dtype=float)

    def forward(self,x):
        return self.L(x.permute(0,2,1)).permute(0,2,1)