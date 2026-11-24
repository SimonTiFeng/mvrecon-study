import sys
from PIL import Image, ImageDraw
src, x0, y0, x1, y1, dst, step = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6], int(sys.argv[7])
im = Image.open(src).convert("RGB").crop((x0, y0, x1, y1)).resize(((x1-x0)*4, (y1-y0)*4), Image.NEAREST)
d = ImageDraw.Draw(im)
W, H = im.size
s = step*4
for x in range(x0 - x0 % step, x1 + 1, step):
    if x < x0: continue
    px = (x - x0)*4
    d.line([(px, 0), (px, H)], fill=(0,255,0), width=1)
    d.text((px+1, 1), str(x), fill=(255,255,0))
for y in range(y0 - y0 % step, y1 + 1, step):
    if y < y0: continue
    py = (y - y0)*4
    d.line([(0, py), (W, py)], fill=(0,255,0), width=1)
    d.text((1, py+1), str(y), fill=(255,255,0))
im.save(dst); print(dst, im.size)
