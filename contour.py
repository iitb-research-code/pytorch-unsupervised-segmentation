import cv2
import numpy as np
import json

img = cv2.imread('output.png', cv2.IMREAD_UNCHANGED)

img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

thresh = 100

ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours[0])

annotation = {
        "segmentation": [],
    }

for contour in contours:
    contour = np.flip(contour, axis=1)
    segmentation = contour.ravel().tolist()
    annotation["segmentation"].append(segmentation)
    
# print(json.dumps(annotation, indent=4))
with open("output.json", "w") as outfile:
    json.dump(annotation, outfile)
cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imwrite('san1c75.png',img)