from albumentations import *
import cv2
import matplotlib.pyplot as plt


# original image
input_image = cv2.cvtColor(cv2.imread('./5245.jpg'), cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.imshow(input_image.astype('uint8'))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('./original_image.png')

gaussian_noise = GaussNoise(var_limit=(100, 500), p=1.0)(image=input_image)['image']
plt.figure(2)
plt.imshow(gaussian_noise.astype('uint8'))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('./gaussian_noise.png')

hor_flip = HorizontalFlip(p=1.0)(image=input_image)['image']
plt.figure(3)
plt.imshow(hor_flip.astype('uint8'))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('./horizontal_flip.png')

rgb_shift = RGBShift(p=1.0)(image=input_image)['image']
plt.figure(4)
plt.imshow(rgb_shift.astype('uint8'))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('./rgb_shift.png')

# channel_shuffle = ChannelShuffle(p=1.0)(image=input_image)['image']
# plt.figure(5)
# plt.imshow(channel_shuffle.astype('uint8'))
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.savefig('./channel_shuffle.png')

ver_flip = VerticalFlip(p=1.0)(image=input_image)['image']
plt.figure(6)
plt.imshow(ver_flip.astype('uint8'))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('./vertical_flip.png')