# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11s.yaml")  # build a new model from YAML
# model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    model = YOLO('yolov8s.yaml')        # 本机的模型配置yaml文件位于./ultralytics/cfg/models/11/yolo11.yaml
    # model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights
    model.train(data = "paint/_paint.yaml",
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=300,
                epochs=300,
                single_cls=False,  # 是否是单类别检测
                batch=512,
                close_mosaic=0,
                workers=0,
                optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                device=[0, 1]
                )
 