from glob import glob
import os
import os.path as osp

errorish_path = "/home/andrea/Documents/project/incorrect images/error/still errorish"

all_imgs_path = "/media/andrea/New Volume/train/train"

imgs = [osp.join(all_imgs_path, osp.basename(x)) for x in glob(osp.join(errorish_path, '*.jpg'))]

[os.remove(x) for x in imgs]