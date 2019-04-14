# Face_Recognition_using_pytorch
Using MTCNN and MobileFaceNet on Face Recognition

This repo is easy to use and convenient to adapt to your own project. You just need one GPU supporting CUDA and do some simple installations, and then you can easily run it successfully.

![image](https://github.com/shoutOutYangJie/Face_Recognition_using_pytorch/blob/master/results/results.gif)

## requirements

PyTorch1.0

torchvision

opencv-python

PIL

numpy

notes that make sure your computer has one camera, because code uses cv2.videoCapture(0) to get image.

## how to use


please going to root dir
firstly for getting your name, run
> python get_save_features.py --name {$YOUR_NAME$}

notes that when cvWindow arises, please left click it to make it being current window, and press 'c' to crop your face, and press 'q' to quit program.

And then you will find your feature vector under the "users/{$YOUR_NAME$}" dir.

secondly, run 
> python verification_multi_face.py 

## acknowledgement
MTCNN[1] comes from [TropComplique](https://github.com/TropComplique/mtcnn-pytorch), that is a great work containing pretrained model.

mobileFaceNet[2] derives from [Xiaoccer](https://github.com/Xiaoccer/MobileFaceNet_Pytorch). I followed this project to train the model on CASIA dataset, but only can get 97.3% on the lfw dataset.

## reference
[1] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
[2] MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices
