
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_rgb = cv2.imread('/Users/g.hoxha/Documents/GitHub/image-processing-app/image-processing-topics/assets/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('/Users/g.hoxha/Documents/GitHub/image-processing-app/image-processing-topics/assets/images/mount2.png', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

plt.imshow(res, cmap='gray')

threshold = 0.8

loc = np.where( res >= threshold)  
for pt in zip(*loc[::-1]):  
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()