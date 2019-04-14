from mtcnn.src import detect_faces, show_bboxes
import torch as t
from ArcFace.mobile_model import mobileFaceNet
from utils import cosin_metric, get_feature, draw_ch_zn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image,ImageFont


def verification():
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

    count = 0

    while ret :
        frame = frame[:,:,::-1]
        img = Image.fromarray(frame)
        bboxes, landmark = detect_faces(img)
        # print(bbox)  # [[296.89171371 211.27569699 441.8924298  396.48678774   0.99999869]]

        if len(bboxes) == 0:
            cv2.imshow('img', frame[:, :, ::-1])
            # videoWriter.write(frame[:,:,::-1])
            cv2.waitKey(10)
            ret, frame = cap.read()
            continue

        show_img = frame.copy()
        for bbox in bboxes:
            loc_x_y = [bbox[2], bbox[1]]
            person_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()
            feature = np.squeeze(get_feature(person_img, model, trans, device))
            cos_distance = cosin_metric(total_features, feature)
            index = np.argmax(cos_distance)
            if not cos_distance[index] > threshold:
                ret, frame = cap.read()
                continue
            person = name_list[index]
            show_img = draw_ch_zn(show_img,person,font,loc_x_y)
            cv2.rectangle(show_img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))


        cv2.imshow('img',show_img[:,:,::-1])
        # videoWriter.write(show_img)

        cv2.imwrite('./results/'+str(count)+'.jpg',show_img[:,:,::-1].copy())
        count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # videoWriter.release()
            break

        ret, frame = cap.read()

if __name__ =='__main__':
    verification()
