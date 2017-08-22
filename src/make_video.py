import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.io import imsave
from skimage.transform import resize
import cv2
import tqdm

def frames2video(name, path):
    """
    Merges images in path into a video

    :param path: path with prediction images
    :return:
    """
    batch_size = 100
    fnames = os.listdir(path)
    fnames.sort()

     
    #images = np.array([plt.imread(os.path.join(path, fname)) for fname in fnames])
    # h, w, c = images[0].shape
    videowriter = imageio.get_writer(name + '_video.mp4', fps=25)

    for fname in tqdm.tqdm(fnames):
        videowriter.append_data(plt.imread(os.path.join(path, fname)))
    videowriter.close()



frames2video('ds2', 'dataset2/pred/')
