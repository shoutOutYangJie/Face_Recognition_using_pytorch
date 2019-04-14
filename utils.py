import numpy as np
from PIL import Image, ImageDraw

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1,axis=1) * np.linalg.norm(x2))

def get_feature(img,model,trans,device):
    img = Image.fromarray(img)
    img = trans(img).to(device)
    embedding = model(img[np.newaxis,:,:,:])
    return embedding.cpu().data.numpy()


def draw_ch_zn(img,str,font,loc):   # for showing Chinese
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((loc[0],loc[1]),str,(255,0,0),font)
    return np.array(pil_img).copy()

# if __name__=='__main__':
#     x1 = np.random.randint(0,10,size=(2,128))
#     x2 = np.random.randn(128,)
#     print(np.dot(x1,x2).shape)
#     print( np.linalg.norm(x1,axis=1).shape)
#     print(cosin_metric(x1,x2).shape)

