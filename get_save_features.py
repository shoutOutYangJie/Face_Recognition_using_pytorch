from ArcFace.mobile_model import mobileFaceNet
from mtcnn.src import detect_faces, show_bboxes
import torch as t
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import os
from utils import get_feature
import argparse


def save_person_information(name):
    saved_model = './ArcFace/model/068.pth'
    info_path = './users/'+name
    if not os.path.exists(info_path):
        os.makedirs(info_path)

    # threshold =  0.30896
    model = mobileFaceNet()
    model.load_state_dict(t.load(saved_model)['backbone_net_list'])
    model.eval()
    use_cuda = t.cuda.is_available() and True
    device = t.device("cuda" if use_cuda else "cpu")
    # is_cuda_avilableqq
    trans = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    model.to(device)

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
        cv2.imshow('img',show_img) # 480 640 3
        if cv2.waitKey(1) & 0xFF == ord('c'):
            person_img = frame[int(bboxes[0,1]):int(bboxes[0,3]),int(bboxes[0,0]):int(bboxes[0,2])]
            cv2.imshow('crop',person_img[:,:,::-1])
            cv2.imwrite(os.path.join(info_path,'%s.jpg'%(name)),person_img[:,:,::-1])
            feature = np.squeeze(get_feature(person_img,model,trans,device))
            np.savetxt(os.path.join(info_path,'%s.txt'%(name)),feature)

            # cv2.waitKey(30)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

if __name__=='__main__':
    parse = argparse.ArgumentParser(description="get current user's face image")
    parse.add_argument('-n','--name',default=None,help='input current user\'s name')
    arg = parse.parse_args()
    name = arg.name
    if name == None:
        raise ValueError('please input your name using \'python get_save_features.py --name your_name\'')
    save_person_information(name)