import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_attention1 = SpatialAttention()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.spatial_attention2 = SpatialAttention()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.spatial_attention3 = SpatialAttention()

        self.fc1 = nn.Linear(2304, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 128)

        self.fc_pose1 = nn.Linear(3,128)
        self.fc_pose2 = nn.Linear(128,128)

    def forward(self, s):
        # print(s.shape)
        s = torch.tensor(s)
        if len(s)==5003:
            s = s.unsqueeze(0)
        x1 = s[:,:50*50]    # First image
        x2 = s[:,50*50:2*50*50]  # Second image
        y = torch.tensor(s[:,2*50*50:])   # The remaining 5 values
        y = y.view(-1,3)

        x1 = torch.tensor(x1).view(-1, 1, 50, 50)
        x2 = torch.tensor(x2).view(-1, 1, 50, 50)
        x = torch.cat((x1, x2), dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        # attention_map1 = self.spatial_attention1(x)
        # x = x * attention_map1
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        # attention_map2 = self.spatial_attention2(x)
        # x = x * attention_map2
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        attention_map3 = self.spatial_attention3(x)
        x = x * attention_map3
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        y = self.relu(self.fc_pose1(y))
        y = self.relu(self.fc_pose2(y))

        output = torch.cat([x, y], axis=-1)

        return output

class ActorNet(nn.Module):
    def __init__(self, state_dim=256, action_dim=5):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.state_dim = state_dim

    def forward(self, x):
        if len(x)==5003:
            x = x.unsqueeze(0)
        x = x.view(-1,self.state_dim)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class CriticNet(nn.Module):
    def __init__(self, state_dim=256):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.state_dim = state_dim

    def forward(self, x):
        if len(x)==5005:
            x = x.unsqueeze(0)
        x = x.view(-1,self.state_dim)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

        
       


