import os, sys

"""
This will add all the root directory path in python env variable
parent methods can be imported from child classes 

*** NEED TO INCLUDE PATH IN __init__.py FILE ***

"""
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import config

label_path = config.BIOVID_VIDEO_LABEL_PATH
image_label = config.BIOVID_PATH + '/' + 'image_labels.txt'

with open(label_path) as file:
    print('Write each image with label using folder text file ...')    

    image_file = open(image_label, "w")
    for line in file:
        
        folder = line.split(' ')
        image_label = folder[1]
        folder_dir = config.BIOVID_PATH + '/' + folder[0]
        
        if not os.path.isdir(folder_dir):
            continue

        for img in os.listdir(folder_dir):
            image_file.write(folder[0] + '/' + img + ' ' + image_label)

    image_file.close()
    print('Done')