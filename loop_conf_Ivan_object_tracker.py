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


# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front\config/yolov3.cfg' #img_size = 416
config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_new_cleaner\config/yolov3.cfg' #img_size = 416
# config_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_truck_car\config/yolov3.cfg' #img_size = 416
# config_path = r'C:\Users\AIC-WS1\Ivan\YoloVideoTrack\config/yolov3_orig.cfg'

# weights_path=r'D:\Ivan\YoloCheckpoints\OID_front_1_erkli_car\checkpoints/yolov3_ckpt_300.pth'
weights_path=r'D:\Ivan\YoloCheckpoints\OID_front_new_cleaner_1_erkli_car\checkpoints/yolov3_ckpt_297.pth'
# weights_path=r'D:\Ivan\YoloCheckpoints\katip_truck_car_416\checkpoints/yolov3_ckpt_302.pth'
# weights_path = r'C:\Users\AIC-WS1\Ivan\YoloVideoTrack\config/yolov3_orig.weights'

# class_path =r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front\config/coco.names'
class_path =r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_new_cleaner\config/coco.names'
# class_path = r'D:\Ivan\Test_data\IvanMadeDataSet\Yolo_front_truck_car\config/coco.names'
# class_path = r'C:\Users\AIC-WS1\Ivan\YoloVideoTrack\config/coco_orig.names'

img_size=416
nms_thres=0.4

carClass = 0

# confArr=[0.9981,0.9982,0.9983,0.9984,0.9985,0.9986,0.9987,0.9988,0.9989]
confArr=[0.9]
videoArr=[]


videoArr = [r'D:/Ivan/Test_data/IvanMadeDataSet/Stanford_AI_cars_modified/car_crop.mp4']

# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200509_161540_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200512_161525_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200515_161547_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200518_161529_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200521_161542_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200530_161507_crop.mp4')
# videoArr.append(r'D:/Ivan/Test_data/Katipunan/test/VID_20200602_161933_crop.mp4')
# videoArr.append(r'D:/Ivan/Test_data/Katipunan/test/VID_20200605_161506_crop.mp4')
# videoArr.append(r'D:/Ivan/Test_data/Katipunan/test/VID_20200611_161534_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200614_161517_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200617_161508_crop.mp4')
# videoArr.append(r'D:/Ivan/Test_data/Katipunan/test/VID_20200620_161507_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200705_161518_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200708_161553_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200711_161516_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200714_161511_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200717_161614_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200720_161556_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200723_161518_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200726_161509_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200729_161519_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200801_161505_crop.mp4')

# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200804_161505_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200807_161502_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200810_161501_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200813_161504_crop.mp4')
# videoArr.append(r'D:\Ivan\Test_data\Katipunan\test/VID_20200816_161502_crop.mp4')

lineX0 = 0
lineX1 = 900
lineY  = 300

print(len(videoArr))

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

for conf_thres in confArr:
    carObjId=[]
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




    for vid in videoArr:
        print(vid)
        videopath = vid 
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
        outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-detModel_2"+"_"+str(conf_thres)+"nms"+str(nms_thres)+".mp4"),fourcc,20.0,(vw,vh))

        frames = 0
        starttime = time.time()
        car_count = 0

        file1 = open(r"D:\Ivan\carsCount.txt","a")
        file1.write(videopath)
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
            cv2.line(frame, (lineX0,lineY),(lineX1,lineY),colors[0],5)
            # cv2.putText(frame, car_count, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
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
                    # cv2.line(frame, (50,100),(600,100),color,5)
                    #if rectangle crosses line i++
                    if(cls_pred==carClass and (obj_id in carObjId)==False and lineY >= y1+box_h ):
                        car_count= car_count+1
                        print(car_count)
                        carObjId.append(obj_id)                    
            cv2.imshow('Stream', frame)
            outvideo.write(frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        totaltime = time.time()-starttime
        print(frames, "frames", totaltime/frames, "s/frame")
        print('number of cars', car_count)
        file1.write(','+str(car_count))
        file1.write(','+ str(conf_thres))
        file1.write('\n')
        file1.close()
        cv2.destroyAllWindows()
        outvideo.release()