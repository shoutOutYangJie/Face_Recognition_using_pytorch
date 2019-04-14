import torch as t
from mobile_model import mobileFaceNet, Arcloss
from load_data import get_train_dataSet, get_eval_dataSet, LFW, parse_list
from torch import nn
from config import get_config
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from img_test import *


# './model/009.pth'
def train(multi_gpu=False,resume=None):
    conf = get_config(is_training=True)
    loader, class_num = get_train_dataSet(conf.data_path,conf)
    identity_list = get_lfw_list(conf.lfw_test_list)
    img_paths = [os.path.join(conf.lfw_root, each) for each in identity_list]
    #nameLs, nameRs, labels = parse_list('lfw')
    # test_dataset = LFW(nameLs,nameRs)
    # test_load#
    backbone = mobileFaceNet()
    arcloss = Arcloss(class_num=class_num)
    # for i in [*backbone.modules()]:
    #     print(i.__class__)
    # for i in arcloss.modules():
    #     print(i)
    ignored_params = list(map(id,backbone.linear1.parameters()))
    ignored_params += list(map(id,arcloss.weight))
    prelu_params = []
    for m in backbone.modules():
        if isinstance(m,nn.PReLU):
            ignored_params += list(map(id,m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params, backbone.parameters())
    optimizer_ft = optim.SGD([
        {'params':base_params,'weight_decay':4e-5},
        {'params':backbone.linear1.parameters(),'weight_decay':4e-4},
        {'params':arcloss.weight,'weight_decay':4e-4},
        {'params':prelu_params,'weight_decay':0.0},
    ],lr=0.1,momentum=0.9,nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,milestones=conf.milestones,gamma=0.1)

    backbone = backbone.cuda()
    arcloss = arcloss.cuda()

    # if multi_gpu:
    #     backbone = nn.DataParallel(backbone)
    #     arcloss = nn.DataParallel(arcloss)

    criterion = conf.loss
    if resume != None:
        ckpt = t.load(resume)
        backbone.load_state_dict(ckpt['backbone_net_list'])
        arcloss.load_state_dict(ckpt['arcloss_net_list'])
        start_epoch = ckpt['epoch']+1
    else:
        start_epoch = 0

    for epoch in range(start_epoch,conf.epochs):
        exp_lr_scheduler.step()
        print('Train Epoch: {}/{}...'.format(epoch,conf.epochs))
        backbone.train()
        arcloss.train()

        train_total_loss = 0.0
        total =0
        since = time.time()
        for data in loader:
            img , label = data[0].cuda(), data[1].cuda()
            batch_size = conf.batch_size
            optimizer_ft.zero_grad()
            raw_logits = backbone(img)
            output = arcloss(raw_logits,label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            train_total_loss +=total_loss.item()*batch_size
            total += batch_size

        train_total_loss = train_total_loss/total
        time_elapsed = time.time()-since
        loss_log = 'total_loss:{:.4f} time: {:.0f}m {:.0f}s'.format(train_total_loss, time_elapsed//60, time_elapsed%60)
        print(loss_log)

        if epoch % conf.save_freq ==0:
            msg = 'Saving checkpoint :{}'.format(epoch)
            print(msg)
            net_state_list = backbone.state_dict()
            arcloss_state_list = arcloss.state_dict()
            t.save({
                'epoch':epoch,
                'backbone_net_list':net_state_list,
                'arcloss_net_list' :arcloss_state_list
            },
            os.path.join(conf.model_path,'%03d.pth'% epoch))

        if epoch % conf.eval_freq ==0:
            backbone.eval()
            # featureLs = None
            # featureRs = None
            # print('Test Epoch: {}...'.format(epoch))
            # for data in test_loader:
            #     for i in range((len(data))):
            #         data[i] = data[i].cuda()
            #     res = [backbone(d).cpu().numpy for d in data]
            acc,th = lfw_test(backbone, img_paths, identity_list, conf.lfw_test_list, conf.test_batch_size)
            with open('record.txt','a') as f:
                f.write('%d epoch, acc is %f ,threshold is %f  \n'%(epoch,acc,th))
    print('training done!')



train()
