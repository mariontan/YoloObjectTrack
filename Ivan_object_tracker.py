from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# config_path = r'D:/Ivan/Test_data/IvanMadeDataSet/OID_Car_35k/Dataset/train/config/yolov3.cfg'
# weights_path = r'D:/Ivan/YoloCheckpoints/OID_Car_35K/15.weights'
# class_path = r'D:/Ivan/Test_data/IvanMadeDataSet/OID_Car_35k/Dataset/train/config/coco.names'

# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_left\config/yolov3.cfg'
# weights_path = r'D:\Ivan\YoloCheckpoints\OID_Left_view_1/29.weights'
# class_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_left\config/coco.names'

config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_new_cleaner\config/yolov3.cfg' #img_size = 416
# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_truck_car\config/yolov3.cfg' #img_size = 416
# config_path = r'C:\Users\AIC-WS1\Ivan\YoloVideoTrack\config/yolov3_orig.cfg'


# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front\config/yolov3_608.cfg' #img_size=608
# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front\config/yolov3_832.cfg' #img_size=832
# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front\config/yolov3_224.cfg' #img_size=224
# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front\config/yolov3_320.cfg' #img_size=320
#weights_path = r'D:\Ivan\YoloCheckpoints\OID_front_1_416_0_9/79.weights'
#weights_path = r'D:\Ivan\YoloCheckpoints\OID_front_2_416_0_9/123.weights'
# weights_path = r'D:\Ivan\YoloCheckpoints\OID_front_1_608/79.weights' #img_size=608
# weights_path = r'D:\Ivan\YoloCheckpoints\OID_front_1_832/80.weights' #img_size=832
# weights_path = r'D:\Ivan\YoloCheckpoints\OID_front_1_224/80.weights' #img_size=224
# weights_path = r'D:\Ivan\YoloCheckpoints\OID_front_1_320/80.weights' #img_size=320

weights_path=r'D:\Ivan\YoloCheckpoints\OID_front_new_cleaner_1_erkli_car\checkpoints/yolov3_ckpt_297.pth'
# weights_path=r'D:\Ivan\YoloCheckpoints\katip_truck_car_416\checkpoints/yolov3_ckpt_302.pth'
# weights_path = r'C:\Users\AIC-WS1\Ivan\YoloVideoTrack\config/yolov3_orig.weights'

class_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_truck_car\config/coco.names'
# class_path = r'C:\Users\AIC-WS1\Ivan\YoloVideoTrack\config/coco_orig.names'
#img_size = 416
#conf_thres = 0.8
#nms_thres = 0.4

img_size=416
#conf_thres=0.99
#conf_thres=0.9995
#conf_thres = 0.998
# conf_thres = 0.998
conf_thres = 0.95
nms_thres=0.4

carClass = 0
# # Load model and weights
# model = Darknet(config_path, img_size=img_size)
# model.load_weights(weights_path)
# model.cuda()
# model.eval()

model = Darknet(config_path, img_size=img_size)

if weights_path.endswith('.weights'):
    model.load_weights(weights_path)
    print('weights file loaded')
else: 
    model.load_state_dict(torch.load(weights_path))
    print('pth file loaded')

model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

# videopath = r'D:/Ivan/Test_data/IvanMadeDataSet/Stanford_AI_cars_modified/car_crop.mp4'
##katipunan data set
# videopath = r'D:\Ivan\Test_data\Katipunan\test/VID_20200509_161540_crop.mp4'
# videopath = r'D:\Ivan\Test_data\Katipunan\test/VID_20200512_161525_crop.mp4'
# videopath = r'D:\Ivan\Test_data\Katipunan\test/VID_20200515_161547_crop.mp4'
videopath = r'D:\Ivan\Test_data\Katipunan\test/VID_20200518_161529_crop.mp4'
# videopath = r'D:\Ivan\Test_data\Katipunan\test/VID_20200518_161529.mp4'
import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-detmodel2_80nms4.mp4"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()
car_count = 1
carObjId = []
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            if(cls_pred==carClass and (obj_id in carObjId)==False ):
                # print('obj_id',obj_id,'cuurentId', carObjId, 'condition',(obj_id in carObjId),'count',car_count )
                car_count= car_count+1
                carObjId.append(obj_id)
            
                
    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
print('number of cars', car_count)
cv2.destroyAllWindows()
outvideo.release()