from PIL import Image
import os, sys

f = sys.argv[1]
im = Image.open(f)
im2 = im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
im2.save("ans2.png","PNG")
