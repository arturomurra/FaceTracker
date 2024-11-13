import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# CNN for image feature extraction
class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Flatten the output

# Multilayer network for spatial location (position and velocity)
class SpatialNN(nn.Module):
    def __init__(self):
        super(SpatialNN, self).__init__()
        self.fc1 = nn.Linear(128 + 2, 256)  # CNN output + position and velocity
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Output for x, y accelerations

    def forward(self, image_features, position_velocity):
        x = torch.cat((image_features, position_velocity), dim=1)  # Concatenate features and position
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # x and y acceleration outputs

# Full model combining CNN and the Spatial NN
class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.cnn = ImageCNN()
        self.spatial_nn = SpatialNN()

    def forward(self, image, position_velocity):
        image_features = self.cnn(image)
        acceleration = self.spatial_nn(image_features, position_velocity)
        return acceleration

# Example usage
model = FullModel()

# Sample input
image = torch.randn(1, 3, 100, 100)  # Batch size 1, RGB image of 100x100
position_velocity = torch.randn(1, 2)  # Position and velocity (x, y)

# Forward pass
acceleration = model(image, position_velocity)
print(acceleration)
