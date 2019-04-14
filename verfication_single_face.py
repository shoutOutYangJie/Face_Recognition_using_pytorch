from mtcnn.src import detect_faces, show_bboxes
import torch as t
from ArcFace.mobile_model import mobileFaceNet
from utils import cosin_metric, get_feature, draw_ch_zn
import numpy as np
import cv2
import os
from torchvision import transforms

from PIL import Image,ImageFont
def verfication():
    saved_model = './ArcFace/model/068.pth'
    name_list = os.listdir('./users')
    path_list = [os.path.join('./users',i,'%s.txt'%(i)) for i in name_list]
    total_features = np.empty((128,),np.float32)
    people_num = len(path_list)

    font = ImageFont.truetype('simhei.ttf',20,encoding='utf-8')

    if people_num>1:
        are = 'are'
        people = 'people'
    else:
        are = 'is'
        people = 'person'
    print('start retore users information, there %s %d %s information'%(are,people_num,people))
    for i in path_list:
        temp = np.loadtxt(i)
        total_features = np.vstack((total_features,temp))
    total_features = total_features[1:]

    # threshold = 0.30896     #阈值并不合适，可能是因为训练集和测试集的差异所致！！！
    threshold = 0.5
    model = mobileFaceNet()
    model.load_state_dict(t.load(saved_model)['backbone_net_list'])
    model.eval()
    use_cuda = t.cuda.is_available() and True
    device = t.device("cuda" if use_cuda else "cpu")

    # is_cuda_avilable
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
    fps = 5
    # videoWriter = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 480))
    while ret :
        frame = frame[:,:,::-1]
        img = Image.fromarray(frame)
        bboxes, landmark = detect_faces(img)
        # print(bboxes)  # [[296.89171371 211.27569699 441.8924298  396.48678774   0.99999869]]

        if len(bboxes) == 0:
            cv2.imshow('img', frame[:, :, ::-1])
            # videoWriter.write(frame[:,:,::-1])
            cv2.waitKey(10)
            ret, frame = cap.read()
            continue

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        area = w*h
        first_person = np.argmax(area)
        bboxes = bboxes[first_person]
        loc_x_y = [bboxes[2],bboxes[1]]
        if bboxes[4]<0.99980:
            temp_img = draw_ch_zn(frame.copy(),'无效的脸部',font,loc_x_y)
            # cv2.putText(temp_img,'valid face ! ', \
            #             (50, 100),
            #             cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2 )
            cv2.rectangle(temp_img, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])),
                          (255, 0, 0))
            cv2.imshow('img', temp_img[:,:,::-1])
            # videoWriter.write(frame[:,:,::-1])
            cv2.waitKey(10)
            ret, frame = cap.read()
            continue

        person_img = frame[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])]
        feature = np.squeeze(get_feature(person_img, model, trans, device))
        cos_distance = cosin_metric(total_features, feature)
        # print(cos_distance)
        index = np.argmax(cos_distance)
        if not cos_distance[index] > threshold:   #  threshold 应该更大点
            temp_img = draw_ch_zn(frame.copy(), '未能匹配当前用户', font, loc_x_y)
            temp_img = draw_ch_zn(temp_img.copy(), '拒绝开锁', font, [loc_x_y[0],loc_x_y[1]+20])
            # cv2.putText(temp_img,'非用户',(50,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
            cv2.rectangle(temp_img, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])),
                          (255, 0, 0))
            cv2.imshow('img',temp_img[:,:,::-1])
            # videoWriter.write(temp_img)
            cv2.waitKey(10)
            ret, frame = cap.read()
            continue
        person = name_list[index]
        show_img = frame.copy()
        show_img = draw_ch_zn(show_img,person,font,loc_x_y)
        cv2.rectangle(show_img,(int(bboxes[0]),int(bboxes[1])),(int(bboxes[2]),int(bboxes[3])),(0,0,255))
        cv2.imshow('img',show_img[:,:,::-1])
        # videoWriter.write(show_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # videoWriter.release()
            break

        ret, frame = cap.read()

if __name__ =='__main__':
    verfication()