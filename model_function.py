import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
from utility_function import Device

def TransferModel(arch):
    model = {
        'VGG':models.vgg16(pretrained=True),
        'AlexNet':models.alexnet(pretrained=True),
        'ResNet':models.resnet18(pretrained=True),
        'DenseNet':models.densenet121(pretrained=True),
    }
    model = model[arch]
    for param in model.parameters():
        param.requires_grad = False

    return model

def GetInputFeatures(model):
    if hasattr(model, "classifier"):
        try:
            input_size = model.classifier[0].in_features
        except TypeError:
            input_size = model.classifier.in_features
    elif hasattr(model, "fc"):
        input_size = model.fc.in_features
    return input_size

def BuildClassifier(arch,train_dataset,hidden_units):
    model = TransferModel(arch)
    input_size = GetInputFeatures(model)
    hidden_sizes = [hidden_units, 256]
    ouput_size = len(train_dataset.classes)
    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(0.5)),
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_sizes[1], ouput_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model

def SetOptimizer(model,learning_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return criterion, optimizer

def SetModel(arch,train_dataset,hidden_units,learning_rate):
    model = BuildClassifier(arch,train_dataset,hidden_units)
    criterion, optimizer = SetOptimizer(model,learning_rate)
    return model, criterion, optimizer
     
def validation(model, testloader, criterion, run_on_gpu):
    test_loss = 0
    accuracy = 0

    model.to(Device(run_on_gpu))
    for images, labels in testloader:

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def validation(model, validloader, criterion, run_on_gpu):
    model.eval()
    valid_loss = 0
    accuracy = 0

    device = Device(run_on_gpu)
    model.to(device)

    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            output = model.forward(images)
            valid_loss += criterion(output, labels).item()
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

def do_deep_learning(model, dataloaders, epochs, print_every, criterion, optimizer, run_on_gpu):
    steps = 0

    device = Device(run_on_gpu)
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                valid_loss, valid_accuracy = validation(model, dataloaders['valid'], criterion, run_on_gpu)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss/len(dataloaders['valid'])),
                      "Validation Accuracy: {:.2f}%".format(100*valid_accuracy/len(dataloaders['valid'])))

                running_loss = 0
                
                # Continue to train
                model.train()

def check_accuracy_on_test(model,testloader,run_on_gpu):
    correct = 0
    total = 0

    device = Device(run_on_gpu)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def SaveCheckPoint(model,arch,dataset,optimizer,learning_rate,hidden_units,save_dir):
    model.eval()
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {
        'model': model,
        'arch': arch,
        'input_size': GetInputFeatures(TransferModel(arch)),
        'output_size': len(dataset.classes),
        'hidden_units':hidden_units,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'class_to_idx': model.class_to_idx,
        'optimizer':optimizer}
    torch.save(checkpoint, save_dir)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    
    return model