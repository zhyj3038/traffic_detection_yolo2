"""
Script to use threading for resizing training images
Speeds up processing time a lot
"""

import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.transform import resize
import os
import threading
import queue

def resize_img(q):
    while not q.empty():
        fname = q.get()
        img = plt.imread('images_orig/'+fname)
        res = resize(img, (416,416,3))
        imsave('images_small/'+fname, res)
        q.task_done()


def main():
    q = queue.Queue()
    num_threads = 20

    for fname in os.listdir('images_orig/'):
        q.put(fname)

    for i in range(num_threads):
        worker = threading.Thread(target=resize_img, args=(q,))
        worker.setDaemon(True)
        worker.start()

    q.join()

if __name__ == "__main__":
    main()
