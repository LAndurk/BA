import cv2
import helperFunctions as hf
import imgProcessing

# labels for different substrates
label_ferrite = [False, False, False, False, False, False, False, False, False, False,
                 False, False, True, True, False, False, False, True, True, True,
                 False, True, False, True, False, False, False, True, False, False,
                 False, False, False, False, False, False, False]

label_sandwich = [False, True, True, False, False, True, True, True, True, True,
                  True, True, True, True, True, True, True, True, True, True,
                  True, False, True, True, False, False, False, False, False, False,
                  False, False, True, True, False, False, False, True, True, False,
                  True]

label_glass = [True, False, False, True, True, False, True, False, False, True,
               True, False, False, True, True, True]

# choose label
label = label_sandwich

# initialize output variables
all = 0
right = 0
falsenegative = 0

# loop through pictures
for i in range(32, 42):  # 32,42  30,38 7,15
    all += 1

    # preprocessing
    path = "sandwich/"
    print("folder: ", path)
    path = ''.join([path, str(i), ".png"])
    img = cv2.imread(path)
    img = hf.resize_and_show("img", img, 4)
    print(img.shape)

    """" lists of parameters
    1, True, 3, 201, 7, 15   #sandwich1
    1, False, 41, 0, 1, 13   #sandwich2
    1, True, 1, 201, 5, 1   #ferrit1
    1, False, 1, 0, 3, 1   #ferrit2
    -1, True, 1, 201, 7, 3   #glas1
    -1, False, 1, 0, 1, 7   #glas2
    """

    segmented_img = imgProcessing.get_segmented_img(img, 1, True, 3, 201, 7, 15)
    path_result = "sandwich/segmented" + str(i) + ".png"
    cv2.imwrite(path_result, segmented_img)

    result, img_output, coordinates = imgProcessing.detect_damage(segmented_img)
    path_result = "sandwich/output" + str(i) + ".png"
    cv2.imwrite(path_result, img_output)

    # produce output
    if result == label[i - 1]:  # detected correct
        right = right + 1
    else:  # detected wrong
	
        # show wrongly detected pictures
        ausgabe = ''.join([str(i), str(label[i-1]), str(result)])
        cv2.imshow(ausgabe, img_output)
		
		# count false negative results
        if label[i - 1]:
            falsenegative += 1


# print out variables
print("number of pictures:", all)
print("detected right:", right)
print("test result:", int(100 * right / all), "%")
print("________")
print("falsepositive:", all - right - falsenegative)
print("falsenegative (Pseudofehler):", falsenegative)

cv2.waitKey(0)
