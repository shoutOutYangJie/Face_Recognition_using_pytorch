from mtcnn.src import detect_faces, show_bboxes
import torch as t
from ArcFace.mobile_model import mobileFaceNet
from utils import cosin_metric, get_feature, draw_ch_zn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image,ImageFont
from glob import glob


def verfication():
    saved_model = './ArcFace/model/068.pth'
    name_list = os.listdir('./users')
    path_list = [os.path.join('./users',i,'%s.txt'%(i)) for i in name_list]
    total_features = np.empty((128,),np.float32)
    people_num = len(path_list)
    font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')

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
    threshold = 0.7
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

    img_current_path = 'xxxxxxxxxx'    #  图像所在目录，写绝对地址

    while True :
        all_imgs = glob(os.path.join(img_current_path, '*.jpg'))
        all_imgs.sort(key=lambda fn: os.path.getmtime(fn))
        img_path = all_imgs[-2]

        frame = cv2.imread(img_path)[:, :, ::-1].copy()

        img = Image.fromarray(frame)
        bboxes, landmark = detect_faces(img)
        # print(bboxes)  # [[296.89171371 211.27569699 441.8924298  396.48678774   0.99999869]]

        if len(bboxes) == 0:
            cv2.imshow('img', frame[:, :, ::-1])
            with open(os.path.join(img_current_path, 'result.txt'), 'r') as f:
                f.write('正常工作中...')
            cv2.imwrite(os.path.join(img_current_path, 'result.jpg'), frame[:, :, ::-1].copy())
            cv2.waitKey(10)
            continue

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        area = w * h
        first_person = np.argmax(area)
        bboxes = bboxes[first_person]
        loc_x_y = [bboxes[2], bboxes[1]]
        if bboxes[4] < 0.99980:
            temp_img = draw_ch_zn(frame.copy(), '无效的脸部', font, loc_x_y)
            cv2.rectangle(temp_img, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])),
                          (0, 0, 255))
            cv2.imshow('img', temp_img[:, :, ::-1])
            with open(os.path.join(img_current_path, 'result.txt'), 'r') as f:
                f.write('请正面面对摄像头，请不要遮挡脸部')
            cv2.imwrite(os.path.join(img_current_path, 'result.jpg'), temp_img[:, :, ::-1].copy())
            # videoWriter.write(frame[:,:,::-1])
            cv2.waitKey(10)
            continue

        person_img = frame[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])]
        feature = np.squeeze(get_feature(person_img, model, trans, device))
        cos_distance = cosin_metric(total_features, feature)
        # print(cos_distance)
        index = np.argmax(cos_distance)
        if not cos_distance[index] > threshold:  # threshold 应该更大点
            temp_img = draw_ch_zn(frame.copy(), '未能匹配当前用户', font, loc_x_y)
            temp_img = draw_ch_zn(temp_img[:, :, ::-1].copy(), '拒绝开锁', font, [loc_x_y[0], loc_x_y[1] + 20])
            cv2.rectangle(temp_img, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])),
                          (0, 0, 255))
            cv2.imshow('img', temp_img[:, :, ::-1])
            with open(os.path.join(img_current_path, 'result.txt'), 'r') as f:
                f.write('未知的用户')
            cv2.imwrite(os.path.join(img_current_path, 'result.jpg'), temp_img[:, :, ::-1].copy())
            cv2.waitKey(10)
            continue
        person = name_list[index]
        show_img = frame.copy()
        show_img = draw_ch_zn(show_img, person, font, loc_x_y)
        cv2.rectangle(show_img, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (255, 0, 0))
        cv2.imshow('img', show_img[:, :, ::-1])
        with open(os.path.join(img_current_path,'result.txt'),'r') as f:
            f.write(person)
        cv2.imwrite(os.path.join(img_current_path,'result.jpg'),show_img[:,:,::-1].copy())
        # videoWriter.write(show_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # videoWriter.release()
            break

if __name__ =='__main__':
    verfication()