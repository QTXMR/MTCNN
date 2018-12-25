from PIL import Image, ImageDraw
import numpy.random as npr

# # annotation
# # 0--Parade/0_Parade_marchingband_1_849 448.51 329.63 570.09 478.23
# annotation = [646, 112, 732, 194]
# im = Image.open(r"E:\XQT\MTCNN-pmb\prepare_data/data/RM-RGB/000001.bmp")
# draw = ImageDraw.Draw(im)
#
# line = 5
# x, y = annotation[0], annotation[1]
# width = annotation[2] - annotation[0]
# height = annotation[3] - annotation[1]
#
# for i in range(1, line + 1):
#     draw.rectangle((x + (line - i), y + (line - i), x + width + i, y + height + i), outline='red')
#
# # imshow(im)
# # show()
# im.save("out.jpeg")

import numpy as np
import cv2

img = cv2.imread(r"E:\XQT\MTCNN-pmb\prepare_data/data/RM-RGB/000001.bmp")
height, width, _ = img.shape
size = npr.randint(12, min(width, height) / 2)
#top_left
nx = npr.randint(0, width - size)
ny = npr.randint(0, height - size)
#random crop
crop_box = np.array([nx, ny, nx + size, ny + size])

cropped_im = img[ny : ny + size, nx : nx + size, :]
resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("resize.jpeg", resized_im)

# img.imshow(resized_im)
# img.show()
