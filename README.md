# YOLO-face: A face detector based on You Only Look Once

Thanks to the official implementation of YOLOv7 https://github.com/WongKinYiu/yolov7

## Prerequisite
* python 3.9 or higher
* Ubuntu 18 or higher

## Data preprocessing

**Update**
* 30/10: WiderFace dataset for yolov7-face: [Google Drive](https://drive.google.com/file/d/1JH6GOM8QDQwNCkHbTej7Fq54B-mWdZ4M/view?usp=share_link)
* 29/10: New annotation for YOLOv7 Face (Unofficial)

**Raw dataset preparation**
[WIDERFACE](http://shuoyang1213.me/WIDERFACE/)

[yolov7-face-label](https://drive.google.com/file/d/1FsZ0ACah386yUufi0E_PVsRW_0VtZ1bd/view?usp=sharing)

Download both the WIDERFACE dataset and the landmarks annotation for yolov7. Then organize your stuffs following the structure below (Images are taken from ```WIDERFACE``` while the equivalent .txt file can be found in ```yolov7-face-label```):
```bat
dataset
|-- WIDERFACE_retina
|---- train
|-------- image_name_abcxyz.jpg
|-------- image_name_abcxyz.txt
|---- val
|-------- image_name_abcxyz.jpg
|-------- image_name_abcxyz.txt
```

**Dataset for official YOLOv5-v7 implementation**

Unfortunately, the official [WIDERFACE Dataset](http://shuoyang1213.me/WIDERFACE/) doesn't have landmarks information. However, landmarks annotation can be found at the repository of [RetinaFace-Pytorch](https://github.com/biubug6/Pytorch_Retinaface). After getting a heap of raw datasets, organize your stuffs following the structure below:
```bat
dataset
|-- WIDERFACE_retina
|---- train
|-------- images
|-------- label.txt
|---- val
|-------- images
|-------- label.txt
```
**Note**: The structure mentioned above is optional.

To create WIDERFACE dataset following YOLOv5-v7 style, just run
```bat
python preprocessing.py
```
Optional Argument:

* ```--save_images```: (boolean) save images
* ```--root_dir```: (string) path to the WIDERFACE_retina dataset (default: dataset/WIDERFACE_retina)
* ```--write_landmark```: (boolean) save annotation having landmark information
* ```--saved_folder```: (boolean) path to the destination (default: dataset/WIDERFACE_yolov5)

You may have to install widerface python package via script:
```bat
pip install python-widerface
```

## Train and test the the YOLOv7 model
The methods of training model are refered from [The offical Implementation of YOLOv7](https://github.com/WongKinYiu/yolov7). An example of training is given in ```training_example.sh``` file (GPU inference is required). 

