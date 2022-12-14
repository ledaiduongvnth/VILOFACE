# WIDER FACE: A Face Detection Benchmark
**Update**
* 30/10: WiderFace dataset for yolov7-face: [Google Drive](https://drive.google.com/file/d/1JH6GOM8QDQwNCkHbTej7Fq54B-mWdZ4M/view?usp=share_link)

**The WIDERFACE dataset**
Link to download: http://shuoyang1213.me/WIDERFACE/

WIDER FACE dataset is a face detection benchmark dataset, of which images are selected from the publicly available WIDER dataset. We choose 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. WIDER FACE dataset is organized based on 61 event classes. For each event class, we randomly select 40%/10%/50% data as training, validation and testing sets. We adopt the same evaluation metric employed in the PASCAL VOC dataset. Similar to MALF and Caltech datasets, we do not release bounding box ground truth for the test images. Users are required to submit final prediction files, which we shall proceed to evaluate.