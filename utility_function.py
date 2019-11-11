import argparse
from torchvision import datasets, transforms
import numpy as np
from os.path import join
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json

def train_get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window for train.py.
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir',type=str,default='checkpoint.pth',help='Set directory to save checkpoints')
    parser.add_argument('--arch',type=str,default='DenseNet',help='Choose architecture. Default=DenseNet')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='Set learning rate. Default=0.001')
    parser.add_argument('--hidden_units',type=int,default=512,help='Set hidden units. Default=512')
    parser.add_argument('--epochs',type=int,default=10,help='Set epochs. Default=10')

    parser.add_argument('--gpu',action='store_true',default=False,help='Run on GPU if available')

    return parser.parse_args()

def predit_get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window for predict.py.
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('input',type=str,default='flowers/valid/91/image_08059.jpg',help='Provide the image path')
    parser.add_argument('checkpoint',type=str,default='checkpoint.pth',help='Provide the checkpoint path')
    parser.add_argument('--topk',type=int,default=5,help='Return top K most likely classes')
    parser.add_argument('--category_names',type=str,default='cat_to_name.json',help='Use a mapping of categories to real names')

    parser.add_argument('--gpu',action='store_true',default=False,help='Run on GPU if available')

    return parser.parse_args()

def LoadData():
	data_dir = r'D:\Flowers'
	train_dir = join(data_dir,'train')
	valid_dir = join(data_dir,'valid')
	test_dir = join(data_dir,'test')

    # Define transforms for the training, validation, and testing sets
	data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(25),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]),
    'test':transforms.Compose([
        transforms.RandomRotation(30),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(join(data_dir,x),data_transforms[x]) for x in ['train','valid','test']}

	dataloaders={
        'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
        'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
    }
    return dataloaders,image_datasets

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an torch tensor.
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    width, height = pil_image.size
    if width>height:
        pil_image.thumbnail([(width/height)*255,255])
    else:
        pil_image.thumbnail([255,(height/width)*255])
    new_w,new_h = pil_image.size
    pil_image = pil_image.crop(((new_w-224)/2,(new_h-224)/2,(new_w+224)/2,(new_h+224)/2))
    np_image = np.array(pil_image)
    np_image = np.divide(np_image,255)
    np_image = np.transpose(np_image)

    return torch.from_numpy(np_image)

def predict(image_path, model, topk, run_on_gpu):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and run_on_gpu==True:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    model.eval()
    images = process_image(image_path)
    images.unsqueeze_(0)

    images = images.to(device,dtype=torch.float)

    with torch.no_grad():
        output = model(images)
    probs, classes_index = output.data.topk(topk)
    probs = torch.exp(probs)
    
    classes = []
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    for index in np.array(classes_index[0]):
        classes.append(str(index_to_class[index]))
    return np.array(probs[0]), classes

def NameConversion(classes,dic_path):
    '''
    Convert label to name using json file provied by user.
    '''
    with open(dic_path, 'r') as f:
        cat_to_name = json.load(f)

    class_name=[]
    for aclass in classes:
        class_name.append(cat_to_name[aclass])
    return class_name

def Device(run_on_gpu):
    '''
    Choose device. CPU or GPU.
    '''
    if torch.cuda.is_available() and run_on_gpu==True:
        return 'cuda'
    else:
        return 'cpu'
    
def Display(ClassName,topk):
    '''
    Display top K class names.
    '''
    for i in range(topk):
        print('Top {}: {}'.format(i+1,ClassName[i]))