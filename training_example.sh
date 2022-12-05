python train.py \
    --workers 8 \
    --batch-size 96 \
    --data data/data.yaml \
    --cfg cfg/yolov7-tiny-face.yaml \
    --name yolov7face \
    --project runs \
    --epochs 300