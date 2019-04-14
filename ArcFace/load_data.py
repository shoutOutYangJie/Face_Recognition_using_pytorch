from torch.utils.data import Dataset ,DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from config import get_config
import scipy.misc
import torch
import numpy as np
import os
# class dataset(Dataset):
#     def __int__(self,root,is_training=True,input_shape=[3,112,112]):
#         super(dataset,self).__init__()
#         self.is_training = is_training
#         self.input_shape = input_shape
#         normalize = T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
#         if self.is_training:
#             self.transfroms = T.Compose([
#                 T.RandomHorizontalFlip(),
#                 T.ToTensor,
#                 normalize
#             ])
#         else:
#             self.transfroms = T.Compose([
#                 T.ToTensor,
#                 normalize
#             ])
#     def __getitem__(self):

def get_train_dataSet(root,conf):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((112,112)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    ds = ImageFolder(root,transform)
    # print('dataset is \n',ds)
    '''
     Dataset ImageFolder
    Number of datapoints: 12880
    Root Location: F:\dataSet\WIDER\WIDER_train\images
    Transforms (if any): Compose(
                             <class 'torchvision.transforms.transforms.RandomHorizontalFlip'>
                             Resize(size=(112, 112), interpolation=PIL.Image.BILINEAR)
                             ToTensor()
                             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                         )
    Target Transforms (if any): None
    '''
    class_num =ds[-1][1]+1
    loader = DataLoader(ds,batch_size=conf.batch_size,shuffle=True,pin_memory=conf.pin_memory,num_workers=0)
    return loader, class_num

def get_eval_dataSet(root,conf):
    transform = T.Compose([
        T.Resize((112,112)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    ds = ImageFolder(root,transform)
    class_num =ds[-1][1]+1
    loader = DataLoader(ds,batch_size=conf.batch_size,shuffle=True,pin_memory=conf.pin_memory,num_workers=0)
    return loader, class_num


class LFW(object):
    def __init__(self,imgl,imgr):
        self.imgl_list = imgl
        self.imgr_list = imgr
    def __getitem__(self, index):
        imgl = scipy.misc.imread(self.imgl_list[index])
        if len(imgl.shape) ==2:
            imgl = np.stack([imgl]*3,axis=2)
        imgr = np.asarray(scipy.misc.imread(self.imgr_list[index]))
        if len(imgr.shape)  ==2:
            imgr = np.stack([imgr]*3 ,axis=2)
        imglist = [imgl, imgr]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i]-127.5)/128.0
            imglist[i] = np.transpose(imglist[i],[2,0,1])
        imgs = [torch.from_numpy(imglist[i]).float() for i in imglist]
        return imgs
    def __len__(self):
        return len(self.imgl_list)


def parse_list(root):
    with open(os.path.join(root,'lfw_test_pair.txt')) as f:
        pairs = f.read().splitlines()
        folder_name = 'lfw-align-128'
        nameLs = []
        nameRs = []
        folds = []
        flags = []
        for i, p in enumerate(pairs):
            p = p.split(' ')
            if len(p) == 3:
                nameL = os.path.join(root, folder_name, p[0])
                nameR = os.path.join(root, folder_name, p[1])
                label = int(p[2])
            nameLs.append(nameL)
            nameRs.append(nameR)
            flags.append(label)
    return  [nameLs, nameRs, label]

if __name__ =='__main__':
    # data_path = 'F:\dataSet\WIDER\WIDER_train\images'
    # conf = get_config(is_training=False)
    # loader ,class_num = get_train_dataSet(data_path,conf)
    # for data, label in loader:
    #     print(data.size())
    #     print(label)
    parse_list('lfw')



'''

torch.Size([32, 3, 112, 112])a
label is tensor([31,  4,  4, 26, 13,  5, 35, 16, 45, 31, 50, 36, 27, 21, 33, 24, 23,  5,
        42, 26, 12, 29, 49, 29, 50, 15, 38,  1,  3, 59, 14,  0])
'''