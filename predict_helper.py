
import torch

from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from collections import OrderedDict
from PIL import Image
import numpy as np
import json

def _load_model(checkpoint_path, gpu):
    model = models.vgg16()
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,1024)),
                            ('relu1', nn.ReLU()),
                            ('drop1', nn.Dropout(p=0.2)),
                            ('fc5', nn.Linear(1024, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier

    if torch.cuda.is_available() and (gpu == True):
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    # print(model)
    # print(checkpoint['class_to_idx'])
    return model, checkpoint

def _process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image)
    #print(pil_image.info)
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    pil_image = test_transforms(pil_image)
    
    np_image = np.array(pil_image)
    
    return np_image

def _get_probs_classes(model, image_path, topk):

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    np_image = _process_image(image_path)
    tc_image = torch.from_numpy(np_image)
    
    # Turn off gradients to speed up this part
    with torch.no_grad():
        #add additional batch dimension to avoid error
        tc_image = torch.from_numpy(np_image).unsqueeze(0)
        #print(torch.from_numpy(np_image).shape)
        #print(tc_image.shape)
        
        logps = model.forward(tc_image)
        ps = torch.exp(logps)
        #print(logps)
        #print(ps.shape)
        probs, classes = ps.topk(topk, dim=1)
        #print(probs.shape)
        return probs, classes

def _print_classes_names(checkpoint, cat_to_name, probs, classes_num, topk):
    # convert tensor -> list type
    np_indexes = classes_num.numpy()
    ls_indexes = np_indexes.tolist()
    np_probs = probs.numpy()
    ls_probs = np_probs.tolist()

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {val: key for key, val in class_to_idx.items()}

    classes_to_probs = {}
    for i in range(topk):
        indice = ls_indexes[0][i]
        class_idx = idx_to_class[indice]
        class_name = cat_to_name[class_idx]

        prob = ls_probs[0][i]
        classes_to_probs[class_name] = prob
    
    print(classes_to_probs)

def _load_mapping_to_name(category_names):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def predict(image_path, checkpoint_path, topk, category_names, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model, checkpoint = _load_model(checkpoint_path, gpu)
    probs, classes_num = _get_probs_classes(model, image_path, topk)
    cat_to_name = _load_mapping_to_name(category_names)
    _print_classes_names(checkpoint, cat_to_name, probs, classes_num, topk)
