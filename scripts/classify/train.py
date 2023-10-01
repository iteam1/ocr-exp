'''
python3 scripts/classify/train.py
'''
import os
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tempfile import TemporaryDirectory

# Init
DIM = 224
batch_size = 16
epochs = 10
learning_rate = 0.0005
dst = 'training'
model_path = os.path.join(dst,'checkpoint.pt')

if not os.path.exists(dst):
    os.mkdir(dst)

transform = transforms.Compose([
    transforms.Resize((DIM,DIM)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

trainset = datasets.ImageFolder('dataset/train',transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

valset = datasets.ImageFolder('dataset/val',transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

classes = trainset.classes
num_classes = len(classes)

print('Total: ',num_classes,'\nclasses: ',classes)
# save classes
with open(os.path.join(dst,'classes.txt'), 'w') as f:
    for cls in classes:
        # write each item on a new line
        f.write("%s\n" % cls)
    print('Done')

dataloders = {'train':trainloader,
                'val':valloader}

dataset_sizes = {'train':len(trainset),'val':len(valset)}

def train_one_epoch(model, optimizer, data_loader, device):

    model.train()

    # Zero the performance stats for each epoch
    running_loss = 0.0
    start_time = time.time()
    total = 0
    correct = 0
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    
        # Print performance statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 batches
            batch_time = time.time()
            speed = (i+1)/(batch_time-start_time)
            print('[%5d] loss: %.3f, speed: %.2f, accuracy: %.2f %%' %
                  (i, running_loss, speed, accuracy))

            running_loss = 0.0
            total = 0
            correct = 0

    # save model
    torch.save(model, model_path)

def test_model(model, data_loader):

    model.eval()

    start_time = time.time()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
            
        print('Finished Testing')
        print('Testing accuracy: %.1f %%' %(accuracy))

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Start training on',device)

    # Use a pre-trained ResNet18
    model = models.resnet18(pretrained=True)

    # Update the fully connected layer based on the number of classes in the dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    # Specity the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("------------------ Training Epoch {} ------------------".format(epoch+1))
        train_one_epoch(model, optimizer, trainloader, device)
        test_model(model, valloader)
    
    torch.save(model, model_path)

    print('Finished Training')