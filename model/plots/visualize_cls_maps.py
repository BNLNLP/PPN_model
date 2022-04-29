# Carlos X. Soto, csoto@bnl.gov, 2022

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

### visualize class maps...

plt.rcParams["figure.figsize"] = (15,15)
imnum = 1
print(img_path[i])

plt.subplot(3, 2, 1)
plt.imshow(gt_cls_map[imnum])
plt.subplot(3, 2, 2)
plt.imshow(F.sigmoid(pred_cls_map.detach()[imnum,0] * -1.))
#plt.imshow(F.sigmoid(pred_cls_map.detach()[imnum,0]))
#plt.imshow(pred_cls_map.detach()[imnum,0])
plt.subplot(3, 2, 3)
plt.imshow(F.sigmoid(pred_cls_map.detach()[imnum,1]))
plt.subplot(3, 2, 4)
plt.imshow(F.sigmoid(pred_cls_map.detach()[imnum,2]))
#plt.show()

plt.figure
plt.subplot(3, 2, 5)
plt.imshow(Image.open(img_path[imnum]))
plt.subplot(3, 2, 6)
plt.imshow(inv_normalize(img[imnum]).permute(1, 2, 0))
plt.show()

[len(b) for b in bars[:10]], [len(b) for b in gt_bars[:10]]