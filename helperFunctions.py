import cv2
from matplotlib import pyplot as plt


def show_histogram(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.plot(histogram)
    plt.show()
    return


def resize_and_show(name, img, scale):
    x = int(img.shape[1] / scale)  # columns
    y = int(img.shape[0] / scale)  # rows
    img = cv2.resize(img, (x, y))
    cv2.imshow(name, img)
    return img
