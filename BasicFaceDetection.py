# Import libraries
import cv2

img = cv2.imread("cr7.png")
img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# initialize and load face detection classifier
pede_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
pede_cascade = cv2.CascadeClassifier()
if not pede_cascade.load(cv2.samples.findFile(pede_cascade_name)):
    print("Error loading xml file")
    exit(0)

# detect faces
found = pede_cascade.detectMultiScale(gray_img, minSize=(20,20))

# how many faces are detected
amount_found = len(found)

# draw a green rectangle around the face
if amount_found != 0:
    for (x,y,width,height) in found:
        cv2.rectangle(img, (x,y), (x+height, y+width), (0,255,0), 3)

cv2.imshow("Image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()