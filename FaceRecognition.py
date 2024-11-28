import os
import cv2
import numpy as np
def load_dataset():

    list = [os.path.join('images', f) for f in os.listdir('images')]
    facesDataset = []
    ids = []
    count = 0

    pede_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    pede_cascade = cv2.CascadeClassifier()
    if not pede_cascade.load(cv2.samples.findFile(pede_cascade_name)):
        print("Error loading xml file")
        exit(0)

    for imagepath in list:
        image = cv2.imread(imagepath)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_np = np.array(gray_image, 'uint8')

        faces = pede_cascade.detectMultiScale(image_np)

        for (x, y, w, h) in faces:
            facesDataset.append(image_np[y:y+h, x:x+w])
            ids.append(count)

        count += 1

    return facesDataset, ids

def train():
    faces, ids = load_dataset()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')

def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    pede_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    pede_cascade = cv2.CascadeClassifier()
    if not pede_cascade.load(cv2.samples.findFile(pede_cascade_name)):
        print("Error loading xml file")
        exit(0)

    image = cv2.imread("images/cr7.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = pede_cascade.detectMultiScale(gray_image, minSize=(20,20))

    for (x, y, w, h) in faces:
        id, accuracy = recognizer.predict(gray_image[y:y+h, x:x+w])

        if(accuracy < 100):
            accuracy = "{0}%".format(round(100-accuracy))
        else:
            id = -1
            accuracy = "{0}%".format(round(100 - accuracy))

        cv2.putText(image, str(id), (x+10, y+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(image, str(accuracy), (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def main():
    recognize_face()


if __name__ == "__main__":
    main()

