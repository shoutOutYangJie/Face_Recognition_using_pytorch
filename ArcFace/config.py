from easydict import EasyDict
import torch
from torch.nn import CrossEntropyLoss
def get_config(is_training):
    conf = EasyDict()
    conf.epochs = 70
    conf.batch_size = 200
    conf.data_path = './CASIA-maxpy-crop'
    conf.model_path = './model'
    conf.is_training = is_training
    conf.lr = 1e-3
    conf.net_mode = 'mobileFaceNet'
    conf.input_size = 112
    conf.lfw_test_list = './lfw/lfw_test_pair.txt'
    conf.lfw_root = './lfw/lfw-align-128'
    conf.embedding_size = 512
    conf.device = torch.device('cuda:0')
    conf.loss = CrossEntropyLoss()
    conf.milestones= [36,48,58]
    conf.pin_memory = True
    conf.save_freq = 1
    conf.eval_freq = 1
    conf.test_batch_size = 128
    return conf
