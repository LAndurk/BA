import cv2
import helperFunctions as hf
import imgProcessing


folder = "D:/Uni/BA/Abgabe/gfk"
picture = "3"

path = ''.join([folder, "/", picture, ".png"])
print(path)
img = cv2.imread(path)
img = hf.resize_and_show("input img", img, 4)
print(img.shape)

"""" lists of parameters
1, True, 7, 201, 7, 5   #sandwich1
1, False, 41, 0, 1, 13   #sandwich2
1, True, 1, 201, 5, 1   #ferrit1
1, False, 1, 0, 3, 1   #ferrit2
-1, True, 1, 201, 7, 3   #glas1
-1, False, 1, 0, 1, 7   #glas2
"""

segmented_img = imgProcessing.get_segmented_img(img, 1, True, 7, 201, 7, 5)
hf.resize_and_show("segmented_img", segmented_img, 1)

result, img_output = imgProcessing.detect_damage(segmented_img)
hf.resize_and_show("img_output", img_output, 1)

cv2.waitKey(0)
