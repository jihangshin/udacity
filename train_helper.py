import torch

from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from collections import OrderedDict
import time
from torch import optim
from workspace_utils import active_session

import error_types as error

def _load_data(train_dir, test_dir='./flowers/test/'):
    #print("_load_data entered : ", train_dir, test_dir)
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

    return train_loader, test_loader, train_data

def _build_model(arch, hidden_units):
    # print("_build_model entered : ", arch, hidden_units)

    model = None
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print("[ERROR] _build_model - Unsupported Arch Option")
        return error.UNSUPPORTED_ARCH_ERROR
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088,hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p=0.2)),
                              ('fc5', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                               ]))
    model.classifier = classifier

    # print('inside _build_model : ')
    # print(model)
    return model

def _train_model(model, train_loader, test_loader, gpu, epochs, learning_rate):
    # print("_train_model entered : ", gpu, epochs, learning_rate)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if (gpu == True) and torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    running_loss = 0
    print_every = 5
    steps = 0

    with active_session():
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the GPU
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(test_loader):.3f}.. "
                          f"Test accuracy: {accuracy/len(test_loader):.3f}")
                    running_loss = 0
                    model.train()
    return model, optimizer
                    
def train(data_directory, arch, hidden_units, epochs, gpu, learning_rate):

    train_loader, test_loader, train_data = _load_data(data_directory)
    model = _build_model(arch, hidden_units)

    if model == error.UNSUPPORTED_ARCH_ERROR:
        return error.UNSUPPORTED_ARCH_ERROR

    model, optimizer = _train_model(model, train_loader, test_loader, gpu, epochs, learning_rate)
  
    return model, train_data, optimizer

def save_model(model, train_data, optimizer, save_dir):

    # TODO: Save the checkpoint
    model.cpu()

    checkpoint = {'state_dict': model.state_dict(),
                'input_size': 25088,
                'output_size': 102,
                'epochs': 1, 
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': train_data.class_to_idx}

    #print(optimizer.state_dict())
    #print(model.state_dict())

    torch.save(checkpoint, save_dir)
