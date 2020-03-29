import torch
import torch.nn as nn
from torch.utils import data
from mds189 import Mds189
import numpy as np
from skimage import io, transform
import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import time

import matplotlib.pyplot as plt

start = time.time()

# Helper functions for loading images.
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def mds_loader(params, is_key_frame=True):
    # Datasets
    # TODO: put the path to your train, test, validation txt files
    if is_key_frame:
        label_file_train =  'dataloader_files/keyframe_data_train.txt'
        label_file_val  =  'dataloader_files/keyframe_data_val.txt'
    else:
        label_file_train = 'dataloader_files/videoframe_data_train.txt'
        label_file_val = 'dataloader_files/videoframe_data_val.txt'
        label_file_test = 'dataloader_files/videoframe_data_test.txt'

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    train_dataset = Mds189(label_file_train,loader=default_loader,transform=transforms.Compose([
                                                transforms.ColorJitter(hue=.05, saturation=.05),
                                                transforms.RandomHorizontalFlip(p=0.33),
                                                transforms.RandomRotation(degrees=15),    
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)
                                            ]))
    train_loader = data.DataLoader(train_dataset, **params)

    val_dataset = Mds189(label_file_val,loader=default_loader,transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)
                                            ]))
    val_loader = data.DataLoader(val_dataset, **params)

    if is_key_frame:
        return train_loader, val_loader

    elif not is_key_frame:
        test_dataset = Mds189(label_file_test,loader=default_loader,transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std)
                                                ]))
        test_loader = data.DataLoader(test_dataset, **params)
        return train_loader, val_loader, test_loader

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(95040, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 8)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # Train the model
    # Loop over epochs
    print('Beginning training..')
    total_step = len(train_loader)
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        # Training
        print('epoch {}'.format(epoch))
        for i, (local_batch,local_labels) in enumerate(train_loader):
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            
            # Forward pass
            outputs = model.forward(local_ims)
            loss = criterion(outputs, local_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (i+1) % 4 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        for i, (local_batch,local_labels) in enumerate(val_loader):
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model.forward(local_ims)
            loss = criterion(outputs, local_labels)
            val_losses.append(loss.item())
        print('finished epoch {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'
                .format(epoch+1, train_losses[-1], val_losses[-1]))

    end = time.time()
    print('Time: {}'.format(end - start))

    # Save the model checkpoint
    torch.save(model.state_dict(), './model/model.ckpt')

    return train_losses, val_losses

def test(model, device, test_loader):
    print('Beginning Testing..')
    with torch.no_grad():
        correct = 0
        total = 0
        predicted_list = []
        groundtruth_list = []
        for (local_batch,local_labels) in test_loader:
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)

            outputs = model.forward(local_ims)
            _, predicted = torch.max(outputs.data, 1)
            total += local_labels.size(0)
            predicted_list.extend(predicted)
            groundtruth_list.extend(local_labels)
            correct += (predicted == local_labels).sum().item()

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

    pl = [p.cpu().numpy().tolist() for p in predicted_list]
    gt = [p.cpu().numpy().tolist() for p in groundtruth_list]


    label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
    for id in range(len(label_map)):
        print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))

def plot_loss(train,val):
    mt = sum(train)/len(train)
    mv = sum(val)/len(val)
    plt.title(" Avg Train Loss: "+str(round(mt,4))+", Avg Val Loss: "+str(round(mv,4)))
    plt.plot([i+1 for i in range(len(train))], train, 'r', label="train")
    plt.plot([i+1 for i in range(len(val))], val, 'b', label="validation")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.jpg")

def main(params, num_epochs, learning_rate, is_train=True, is_key_frame=True):
    
    model_to_load = './model/model.ckpt' 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")
    if is_key_frame:
        train_loader, val_loader = mds_loader(params, is_key_frame=is_key_frame)
    if not is_key_frame:
        train_loader, val_loader, test_loader = mds_loader(params, is_key_frame=is_key_frame)

    model = NeuralNet().to(device)

    if is_train:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_losses, val_losses = train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs)
        plot_loss(train_losses, val_losses)
    if not is_train:
        num_epochs = 0
        model.load_state_dict(torch.load(model_to_load))
        test(model, device, val_loader)

if __name__ == "__main__":
    is_train=False
    is_key_frame=True
    params = {'batch_size': 128,
            'shuffle': True,
            'num_workers': 4
            }
    num_epochs = 10
    learning_rate = 1e-4
    main(params, num_epochs, learning_rate, is_train, is_key_frame)