import cv2
import numpy as np

img = cv2.imread("images/cr7.png")
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) # Use fx and fy to resize img

# 1- Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2- Edges
edges = cv2.Canny(gray_img, 150, 300)
img[edges==255] = (255,0,0)

# 3- Corners
dst = cv2.cornerHarris(gray_img, 3, 5, 0.1)

corners = dst > 0.05 * dst.max() # boolean bitmap for corners

coord = np.argwhere(corners)

# mark corners by dots
for y, x in coord:
    cv2.circle(img, (x, y), 3, (0,0,255), -1)

# 3- Laplacian of Gaussian: Regions with high intensity change
# create window
cv2.namedWindow("Laplacian", cv2.WINDOW_AUTOSIZE)

ddepth = cv2.CV_16S
kernel_size = 3
blobdst = cv2.Laplacian(gray_img, ddepth, ksize=kernel_size)

# convert to uint8
abs_blobdst = cv2.convertScaleAbs(blobdst)

cv2.imshow("image", img)
cv2.imshow("Laplacian", abs_blobdst)

cv2.waitKey(0)

cv2.destroyAllWindows()