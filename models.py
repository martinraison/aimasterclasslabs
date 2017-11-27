import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28*28, 27)

    def forward(self, x):
        x = self.fc0(x.view(x.size(0), -1))
        
        return F.log_softmax(x)
