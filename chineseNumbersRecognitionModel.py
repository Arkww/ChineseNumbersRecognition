import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import kagglehub
import torch.optim.lr_scheduler as lrSched
from ChineseMNISTdataset import ChineseMNISTdataset
from NeuralNetwork import NeuralNet
from trainModel import train_model
from computeAccuracy import computeAccuracy


#VARIABLES

#Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#All the chinese numbers
class_names = ['0'  , '1'  , '2'  , '3'  , '4'  , '5'  , '6'  , '7'  , '8'  , '9'  , '10' , '100', '1000', '10000', '100000000']

#Dataset
path = kagglehub.dataset_download("gpreda/chinese-mnist")

#The path for the images files of the dataset
IMAGE_PATH = path + "/data/data/"

BATCH_SIZE = 64
NUM_EPOCHS = 10
lossHistory = []


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is in 1 channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Use single values for grayscale
])


# Preparing the data
dataset = pd.read_csv('chinese_mnist.csv')
dataset['file'] = dataset[['suite_id','sample_id','code']].apply(lambda x: 'input_' + x['suite_id'].astype(str) +'_'+x['sample_id'].astype(str)+'_'+x['code'].astype(str)+'.jpg', axis=1)
chineseNumbers = ChineseMNISTdataset(dataset, IMAGE_PATH)  # init


#Separate into train and test datasets
train_size = int(0.8 * len(chineseNumbers))  # 80% for training
test_size = len(chineseNumbers) - train_size  # Remaining 20% for testing
train_data, test_data = random_split(chineseNumbers, [train_size, test_size])

train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=True)



#Initialize the neural network
net = NeuralNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = lrSched.StepLR(optimizer, step_size = 30, gamma = 0.1)


#Training the model with training data
loss_history = train_model(net, train_loader, optimizer, scheduler, loss_function, NUM_EPOCHS, device, class_names, BATCH_SIZE)
plt.plot(lossHistory)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
plt.show()



#Testing with the test data and displaying the confusion matrix
acc = computeAccuracy(net, test_loader,class_names)
print(f"\nAccuracy: {acc*100} %")

#Saving the model
torch.save(net.state_dict(), 'trained_net.pth')