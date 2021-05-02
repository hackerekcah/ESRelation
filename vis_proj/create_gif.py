from glob import glob
import os
import subprocess
import shutil
from shutil import copy
VIS_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def images2gif():
    subdirs = glob("{}/[!_]*/".format(VIS_ROOT))
    for d in subdirs:
        label = d.split('/')[-2]
        images = os.path.join(d, "{}_relation_grid_%d.png".format(label))
        video_dir = os.path.join(d, "{}.avi".format(label))
        subprocess.run("ffmpeg -y -f image2 -r 2 -i {} {}".format(images, video_dir).split(" "))
        gif_dir = os.path.join(d, "{}.gif".format(label))
        subprocess.run("ffmpeg -y -i {} -pix_fmt rgb24 {}".format(video_dir, gif_dir).split(" "))

def copy2gh():
    gifs = glob("{}/*/*.gif".format(VIS_ROOT))
    wavs = glob("{}/*/*.wav".format(VIS_ROOT))
    pngs = glob("{}/*/*structure.png".format(VIS_ROOT))
    dst_dir = "{}/gh-pages/assets/gifs/".format(PROJ_ROOT)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for f in gifs:
        fname = os.path.basename(f)
        copy(f, os.path.join(dst_dir, fname))

    dst_dir = "{}/gh-pages/assets/wavs/".format(PROJ_ROOT)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for f in wavs:
        fname = os.path.basename(f)
        copy(f, os.path.join(dst_dir, fname))

    dst_dir = "{}/gh-pages/assets/pngs/".format(PROJ_ROOT)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for f in pngs:
        fname = os.path.basename(f)
        copy(f, os.path.join(dst_dir, fname))


if __name__ == '__main__':
    images2gif()
    copy2gh()