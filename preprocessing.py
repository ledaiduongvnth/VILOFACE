"""
This code is used for processing WIDERFACE dataset provided by https://github.com/biubug6/Pytorch_Retinaface 
Download the datset and organize it following the tree below
WIDERFACE_retina
--train
----images
----labels.txt
--val
----images
----labels.txt
"""

from distutils.command import check
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from wider import WIDER # pip install python-widerface
import subprocess
import shutil
import argparse

from utils.wider_face import WiderFaceDetection

WIDERFACE = WiderFaceDetection(txt_path="dataset/WIDERFACE_retina/train/label.txt")

def draw_annotation(img, bbox, lms, exist_lms):
    """ 
    Draw bounding box and landmarks 
    Note: bbox in (x1, y1, x2, y2) style
    """
    r_bgr = (0, 0, 255)
    g_bgr = (0, 255, 0)
    points = []
    for i in range(0, 9, 2):
        x = int(lms[i])
        y = int(lms[i+1])
        points.append((x, y))
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    bgr = cv2.rectangle(img, start_point, end_point, g_bgr, 2)
    if exist_lms:
        for lm in points:
            bgr = cv2.circle(bgr, lm, 2, r_bgr, 2)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def yolobbox2bbox(x,y,w,h):
    """Convert YOLO bbox to x1y1x2y2 style"""
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

def bbox2yolobbox(size, bbox):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (bbox[0] + bbox[2])/2.0
    y = (bbox[1] + bbox[3])/2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h) # opencv imread return (y, x) style

def facelm2cocokeypoint(landmarks = None,
                        keep_coco_style = False):
    """ Convert facial landmarks to COCO keypoints style """
    new_landmarks = [0 if int(x) == -1 else x for x in list(landmarks)]
    insert_indexes = [2, 5, 8, 11, 14]
    for idx in insert_indexes:
        new_landmarks.insert(idx, 2.0)
    if keep_coco_style:
        padding = list(np.zeros(34-len(new_landmarks)))
        new_landmarks += padding
    return new_landmarks


def check_folder(path):
    """ Create folder if it doesn't exist! """
    if not os.path.exists(path):
        os.makedirs(path)


def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return (x1, x2, y1, y2)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_val_set(root: str = None, 
                    ignore_small: int = 0,
                    saved_folder: str = None,
                    write_landmark: bool = True,
                    save_images: bool = True):
    """ convert widerface datset from retina to yolov5 style version 1.0 (no landmark)"""
    phase = "val"
    print("Creating {} dataset".format(phase))
    assert root is not None
    assert saved_folder is not None

    train_images_path = os.path.join(saved_folder, "train/images")
    valid_images_path = os.path.join(saved_folder, "valid/images")
    train_labels_path = os.path.join(saved_folder, "train/labels")
    valid_labels_path = os.path.join(saved_folder, "valid/labels")
    check_folder(saved_folder)
    check_folder(train_images_path)
    check_folder(valid_images_path)
    check_folder(train_labels_path)
    check_folder(valid_labels_path)
    check_datayaml(saved_folder)

    datas = {}
    with open('{}/{}/label.txt'.format(root, phase), 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if '#' in line:
                path = '{}/{}/images/{}'.format(root, phase, line.split(' ')[1])
                img = cv2.imread(path)
                height, width, _ = img.shape
                datas[path] = list()
            else:
                box = np.array(line.split()[0:4], dtype=np.float32)  # (x1,y1,w,h)
                if box[2] < ignore_small or box[3] < ignore_small:
                    continue
                box = convert((width, height), xywh2xxyy(box))
                if write_landmark:
                    label = '0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'.format(box[0], box[1], box[2], box[3])
                else:
                    label = '0 {} {} {} {}'.format(box[0], box[1], box[2], box[3])
                datas[path].append(label)

        set_path = os.path.join(saved_folder, "valid")
        for _, data in tqdm(enumerate(datas.keys())):
            pict_name = os.path.basename(data)
            out_txt = '{}/{}.txt'.format(os.path.join(set_path, "labels"), os.path.splitext(pict_name)[0])
            if save_images:
                img = cv2.imread(data)
                saved_img_path = os.path.join(set_path, "images")+'/'+pict_name
                cv2.imwrite(saved_img_path, img)
            labels = datas[data]
            f = open(out_txt, 'w')
            for label in labels:
                f.write(label + '\n')
        f.close()
    return datas


def check_datayaml(saved_folder: str = None):
    if os.path.exists(os.path.join(saved_folder, "data.yaml")):
        datayaml = "train: "+os.path.join(saved_folder, "train/images")+'\n' \
                    + "val: "+os.path.join(saved_folder, "valid/images")+'\n' \
                    + '\n' \
                    + "nc: 1\n" \
                    + "names: ['faces']"
        with open(os.path.join(saved_folder, "data.yaml"), 'w') as f:
            f.write(datayaml)


def convert_train_set(widerface: WiderFaceDetection = None,
                             write_landmark: bool = True,
                             saved_folder: str = "./dataset",
                             save_image: bool = False):
    """ convert widerface datset from retina to yolov5 style"""
    assert widerface is not None
    assert saved_folder is not None
    check_folder(saved_folder)
    check_datayaml(saved_folder)
    
    # Create train and valid folders
    train_images_path = os.path.join(saved_folder, "train/images")
    valid_images_path = os.path.join(saved_folder, "valid/images")
    train_labels_path = os.path.join(saved_folder, "train/labels")
    valid_labels_path = os.path.join(saved_folder, "valid/labels")
    check_folder(train_images_path)
    check_folder(valid_images_path)
    check_folder(train_labels_path)
    check_folder(valid_labels_path)

    aa=widerface
    for i in tqdm(range(len(aa.imgs_path))):
        img = cv2.imread(aa.imgs_path[i])
        base_img = os.path.basename(aa.imgs_path[i])
        base_txt = os.path.basename(aa.imgs_path[i])[:-4] +".txt"
        save_img_path = os.path.join(train_images_path, base_img)
        save_txt_path = os.path.join(train_labels_path, base_txt)
        with open(save_txt_path, "w") as f:
            height, width, _ = img.shape
            labels = aa.words[i]
            if len(labels) == 0:
                continue
            for _, label in enumerate(labels):
                annotation = np.zeros((1, 14))
                # bbox
                label[0] = max(0, label[0])
                label[1] = max(0, label[1])
                label[2] = min(width -  1, label[2])
                label[3] = min(height - 1, label[3])
                annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
                annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
                annotation[0, 2] = label[2] / width  # w
                annotation[0, 3] = label[3] / height  # h
                #if (label[2] -label[0]) < 8 or (label[3] - label[1]) < 8:
                #    img[int(label[1]):int(label[3]), int(label[0]):int(label[2])] = 127
                #    continue
                # landmarks
                annotation[0, 4] = label[4] / width  # l0_x
                annotation[0, 5] = label[5] / height  # l0_y
                annotation[0, 6] = label[7] / width  # l1_x
                annotation[0, 7] = label[8]  / height # l1_y
                annotation[0, 8] = label[10] / width  # l2_x
                annotation[0, 9] = label[11] / height  # l2_y
                annotation[0, 10] = label[13] / width  # l3_x
                annotation[0, 11] = label[14] / height  # l3_y
                annotation[0, 12] = label[16] / width  # l4_x
                annotation[0, 13] = label[17] / height  # l4_y
                str_label="0 "
                if not write_landmark:
                    annotation_length = 4
                else:
                    annotation_length = len(annotation[0])
                for i in range(annotation_length):
                    str_label =str_label+" "+str(annotation[0][i])
                str_label = str_label.replace('[', '').replace(']', '')
                str_label = str_label.replace(',', '') + '\n'
                f.write(str_label)
        if save_image:
            cv2.imwrite(save_img_path, img)


def convert_retina_to_yolov7(widerface: WiderFaceDetection = None,
                             saved_folder: str = "dataset/WIDERFACE_yolov7",
                             widerface_orig_path: str = "dataset/WIDERFACE",
                             download_widerface_val: bool = False,
                             write_landmarks: bool = True,
                             save_images: bool = False,
                             create_trainset: bool = True,
                             create_valset: bool = True):
    """ 
    widerface (WiderFaceDetection): WIDER Face dataset from RetinaFace-Pytorch
    saved_folder (string): path to the saved location
    write_landmarks (boolean): create training set with label having landmarks
    """
    assert widerface is not None
    check_folder(saved_folder)
    check_datayaml(saved_folder)
    
    # Create train and valid folders
    train_images_path = os.path.join(saved_folder, "train/images")
    valid_images_path = os.path.join(saved_folder, "valid/images")
    train_labels_path = os.path.join(saved_folder, "train/labels")
    valid_labels_path = os.path.join(saved_folder, "valid/labels")
    check_folder(train_images_path)
    check_folder(valid_images_path)
    check_folder(train_labels_path)
    check_folder(valid_labels_path)

    # Create train dataset
    if create_trainset:
        print("Creating training dataset...")
        for i, (image, labels) in tqdm(enumerate(widerface)):
            image_name = str(i)+".jpg"
            label_name = str(i)+".txt"
            textfile = open(os.path.join(train_labels_path, label_name), "w")

            for label in labels:
                # Convert bbox to coco bbox style
                im_size = image.size()
                new_bbox = list(bbox2yolobbox((im_size[1], im_size[0]), label[:4])) 
                # Convert lms to coco keypoints style
                if write_landmarks:
                    new_landmarks = facelm2cocokeypoint(landmarks = label[4:14], 
                                                        keep_coco_style = False) 
                else:
                    new_landmarks = []
                c = [0] # Class index
                new_label = c + new_bbox + new_landmarks
                # Write to annotation file
                line = str(new_label) + "\n"
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.replace(',', '')
                textfile.write(line)
            textfile.close()
            # Write image to file
            if save_images:
                cv2.imwrite(os.path.join(train_images_path, image_name), image.numpy())

    if create_valset:
    # Create validation dataset
        check_folder(widerface_orig_path)
        if download_widerface_val: # it is still unstable, shouldn't use it
            subprocess.call(['sh', './download_widerface_val.sh'])
        wider = WIDER(os.path.join(widerface_orig_path, "wider_face_split"),
                      os.path.join(widerface_orig_path, "WIDER_val/images"),
                      "wider_face_val.mat")
        print("Creating validation dataset...")
        for data in tqdm(wider.next()):
            image_name = os.path.basename(data.image_name)
            label_name = image_name.replace("jpg", "txt")
            image = cv2.imread(data.image_name)
            im_size = image.shape
            bboxes = data.bboxes
            if write_landmarks:
                lms = list(np.full(34, 0.0))
            else:
                lms = []
            c = [0] # class index
            labels = [(c+list(bbox2yolobbox((im_size[1], im_size[0]), bbox))+lms) for bbox in bboxes]
            textfile = open(os.path.join(valid_labels_path, label_name), "w")
            for label in labels:
                line = str(label) + "\n"
                line = line.replace('[', '').replace(']', '')
                line = line.replace(',', '')
                textfile.write(line)
            textfile.close()
            if save_images:
                cv2.imwrite(os.path.join(valid_images_path, image_name), image)


def convert_retina_to_yolov5(root: str = "dataset/WIDERFACE_retina",
                             write_landmark: bool = False,
                             saved_folder: str = "dataset/WIDERFACE_yolov5",
                             save_images: bool = True):
    train_label_txt = os.path.join(root, "train/label.txt")
    widerface = WiderFaceDetection(txt_path=train_label_txt)

    convert_train_set(widerface, write_landmark, saved_folder, save_images),
    convert_val_set(root, write_landmark, saved_folder, save_images),

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition maybe?
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="dataset/WIDERFACE_retina",
                        help="Path to the WIDERFACE_retina dataset")
    parser.add_argument("--write_landmark", action='store_true',
                        help="Create dataset with landmarks")
    parser.add_argument("--save_images", action='store_true',
                        help="Save image")
    parser.add_argument("--saved_folder", type=str, default="dataset/WIDERFACE_yolov5",
                        help="Path to the destination")
    args = parser.parse_args()                                 
    # convert_retina_to_yolov7(widerface=WIDERFACE, 
    #                          create_trainset=True, 
    #                          write_landmarks = False,
    #                          create_valset = True,
    #                          save_images = False)

    convert_retina_to_yolov5(root=args.root_dir,
                             write_landmark=args.write_landmark,
                             saved_folder=args.saved_folder,
                             save_images=args.save_images)