import time

import numpy as np
import matplotlib.pyplot as plt

import sys



def show_image(image_num=1):
    fp = 'saved_images/saved_image_{}.png.npy'.format(image_num)
    loaded_obs = np.load(fp)

    imgplot = plt.imshow(loaded_obs)
    plt.show()

    #np.save(fp, observation)

show_image(1)
show_image(2)
show_image(3)
show_image(4)
show_image(5)
show_image(6)
show_image(7)
