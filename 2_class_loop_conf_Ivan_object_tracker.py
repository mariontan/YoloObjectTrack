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

def formatPaths(path) :
    return path[:-1].split('=')[1].replace('/','\\')

def readTextFile():
    f = open(sys.argv[-1],"r") #read from text file the configratins
    lines = f.readlines()
    videoFiles=[]
    config_path = formatPaths(lines[0])
    weights_path = formatPaths(lines[1])
    class_path = formatPaths(lines[2])
    confArr = lines[3].split('=')[1][:-1].split(',')
    confArr = [float(i) for i in confArr]
    files = lines[4:]
    for line in files:
        video = line[:-1].replace('/','\\')
        videoFiles.append(video)
    return config_path, weights_path, class_path, confArr, videoFiles

config_path, weights_path, class_path,confArr,videoArr = readTextFile()

img_size=416
nms_thres=0.4

carClass = 0
trkClass = 1


lineX0 = 0
lineX1 = 1800
lineY  = 300

print('videoArr len',len(videoArr))
modelLoadStart = time.time()
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
modelLoadEnd = time.time()-modelLoadStart

for conf_thres in confArr:
    carObjId=[]
    trkObjId=[]
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
        starttime = time.time()
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
        outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-detModel_5"+"_"+str(conf_thres)+"nms"+str(nms_thres)+".mp4"),fourcc,20.0,(vw,vh))

        frames = 0
        car_count = 0
        trk_count = 0

        file1 = open(r"D:\Ivan\carsTrkCount.txt","a")
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
            cv2.putText(frame, str(car_count), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 3)
            cv2.putText(frame, str(trk_count), (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 3)
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
                    if(cls_pred==carClass and (obj_id in carObjId)==False and lineY <= y1+box_h ):
                        car_count= car_count+1
                        # print(car_count)
                        carObjId.append(obj_id)
                    if(cls_pred==trkClass and (obj_id in trkObjId)==False and lineY <= y1+box_h ):
                        trk_count= trk_count+1
                        # print(trk_count)
                        trkObjId.append(obj_id)                    
            cv2.imshow('Stream', frame)
            outvideo.write(frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        totaltime = time.time()-starttime
        print(frames, "frames", totaltime/frames, "s/frame")
        print('number of cars', car_count)
        print('number of cars', trk_count)
        file1.write(','+str(car_count))
        file1.write(','+str(trk_count))
        file1.write(','+ str(conf_thres))
        file1.write(','+ str(frames))#frames
        file1.write(','+ str(totaltime))#seconds
        file1.write(','+ str(modelLoadEnd))#modelLoadTIme
        file1.write('\n')
        file1.close()
        cv2.destroyAllWindows()
        outvideo.release()