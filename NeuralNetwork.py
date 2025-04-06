import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Output: (64, 64, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (128, 64, 64)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: (256, 64, 64)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Output: (512, 64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces each dimension by half

        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 15)  # Output layer with 15 classes

        
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolution layers and fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> BatchNorm -> ReLU -> Pool
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))  # Fully connected layer 1 with dropout
        x = self.dropout(F.relu(self.fc2(x)))  # Fully connected layer 2 with dropout
        x = self.fc3(x)  # Fully connected layer 3 (output)
        return x


