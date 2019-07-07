# myls.py
# Import the argparse library
import argparse

import os
import sys

import train_helper 
import error_types as error

# Create the parser
my_parser = argparse.ArgumentParser(description='Train CLI')
#my_parser.add_argument('--input', action='store', type=int, required=True)
#my_parser.add_argument('--id', action='store', type=int)

# Add the arguments
my_parser.add_argument('dataDirectory',
                       metavar='data_directory',
                       type=str,
                       help='the path where data is placed')
my_parser.add_argument('--save_dir', action='store', type=str, default='./checkpoint.pth')
my_parser.add_argument('--arch', action='store', type=str, default='vgg16')
my_parser.add_argument('--learning_rate', action='store', type=float, default=0.002)
my_parser.add_argument('--hidden_units', action='store', type=int, default=1024)
my_parser.add_argument('--epochs', action='store', type=int, default=1)
my_parser.add_argument('--gpu', action='store_true')

# Execute the parse_args() method 
args = my_parser.parse_args()

input_path = args.dataDirectory
save_dir = args.save_dir
arch = args.arch
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu
learning_rate = args.learning_rate
print(args)

if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()

model, train_data, optimizer = train_helper.train(input_path, arch, hidden_units, epochs, gpu, learning_rate)

if model == error.UNSUPPORTED_ARCH_ERROR:
    print("[ERROR] Unsupported arch is entered")
else:
    train_helper.save_model(model, train_data, optimizer, save_dir)
