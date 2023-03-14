from PIL import Image
import os, sys
import glob

path = "/Users/scottysingh/Downloads/birds/test/"
newpath = "/Users/scottysingh/Desktop/birds-384/test/"
dirs = os.listdir( path )

IMAGE_SIZE = 384

def resize():
    for filename in glob.iglob(path + '**/*.jpg', recursive=True):
        im = Image.open(filename)
        f, e = os.path.splitext(filename)
        dirname = os.path.basename(os.path.dirname(filename))
        basename = os.path.basename(filename)
        print(filename)
        imResize = im.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
        new_dir = newpath + dirname + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        if imResize.mode in ("RGBA", "P"):
            imResize = imResize.convert("RGB")
        try:
            imResize.save(newpath + dirname + '/' + basename, 'JPEG')
        except e:
            print(e)
        

resize()