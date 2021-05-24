# Read tags from CVAT labeling, organise images with no labels in 'background' and 'test'

import argparse
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import shutil


def get_img_list(tr, label):
    img_list = []
    for elem in tr:
        for e in elem:
            if e.tag == 'tag' and e.attrib['label'] == label:
                img_list.append(elem.attrib['name'])
    return img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_file', type=str,
                        help='xml to read')
    parser.add_argument('img_src', type=str,
                        help='Location of the images')
    parser.add_argument('img_dst', type=str,
                        help='folder where the images will be copied to')
    parser.add_argument('label', type=str,
                        help='tag of images to copy')

    param = parser.parse_args()

    tree = (ET.parse(param.xml_file))
    root = tree.getroot()

    im_list = get_img_list(root, param.label)
    print('Found {} images with label {}'.format(len(im_list), param.label))

    os.makedirs(param.img_dst, exist_ok=True)

    for img in tqdm(im_list):
        shutil.copy(os.path.join(param.img_src, img), os.path.join(param.img_dst, img))

