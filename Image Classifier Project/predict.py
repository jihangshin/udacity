# myls.py
# Import the argparse library
import argparse

import os
import sys

import predict_helper 

# Create the parser
my_parser = argparse.ArgumentParser(description='Predict CLI')

# Add the arguments
my_parser.add_argument('path_to_image',
                       metavar='path_to_image',
                       type=str,
                       help='the path_to_image where image is placed')
my_parser.add_argument('checkpoint_path',
                       metavar='checkpoint_path',
                       type=str,
                       help='the checkpoint_path where model is saved')

my_parser.add_argument('--top_k', action='store', type=int, default=5)
my_parser.add_argument('--category_names', action='store', type=str, default='./cat_to_name.json')
my_parser.add_argument('--gpu', action='store_true')

# Execute the parse_args() method 
args = my_parser.parse_args()

path_to_image = args.path_to_image
checkpoint_path = args.checkpoint_path
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu
print(args)

if not os.path.exists(path_to_image):
    print('The path_to_image specified does not exist')
    sys.exit()
if not os.path.exists(checkpoint_path):
    print('The checkpoint_path specified does not exist')
    sys.exit()

predict_helper.predict(path_to_image, checkpoint_path, top_k, category_names, gpu)