from ArcFace.mobile_model import mobileFaceNet
from mtcnn.src import detect_faces, show_bboxes
import torch as t
from PIL import Image
import numpy as np
import cv2
saved_model = './ArcFace/model/068.pth'
threshold =  0.30896
model = mobileFaceNet()
model.load_state_dict(t.load(saved_model)['backbone_net_list'])
model.eval()
# is_cuda_avilable

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('failed open camara!!!')
ret, frame = cap.read()
while ret :
    frame = frame[:,:,::-1]
    img = Image.fromarray(frame)
    bboxes, landmark = detect_faces(img)
    show_img = show_bboxes(img,bboxes,landmark)
    show_img = np.array(show_img)[:,:,::-1]
    cv2.imshow('img',show_img)
    cv2.waitKey(30)
    ret, frame = cap.read()