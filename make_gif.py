import imageio
from glob import glob
import os
import cv2


img_list = glob('results/*.jpg')

img_list.sort(key= lambda x: os.path.getmtime(x))
images = []
for img in img_list:
    images.append(imageio.imread(img))
imageio.mimsave('results.gif', images, fps=5)


# fps = 7
# videoWriter = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 480))
# for img in img_list:
#     i = cv2.imread(img)
#     videoWriter.write(i)
# videoWriter.release()