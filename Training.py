import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont

# load test images
ran_faces = os.listdir("testimages/")

# train on these images
cristiano = face_recognition.load_image_file("images/cristiano.png")
cristiano_ec = face_recognition.face_encodings(cristiano)[0]
messi = face_recognition.load_image_file("images/messi.png")
messi_ec = face_recognition.face_encodings(messi)[0]
michael_jordan = face_recognition.load_image_file("images/michael jordan.png")
michael_jordan_ec = face_recognition.face_encodings(michael_jordan)[0]

# identify faces
known_face_encodings = [
    cristiano_ec,
    messi_ec,
    michael_jordan_ec,
]

# faces names
known_face_names = [
    "Cristiano Ronaldo",
    "Lionel Messi",
    "Michael Jordan",
]

for ranface in ran_faces:
    ranimage = face_recognition.load_image_file(f"testimages/{ranface}")
    ranface_locations = face_recognition.face_locations(ranimage)
    ranface_encodings = face_recognition.face_encodings(ranimage, ranface_locations)

    # convert to PIL format
    pil_image = Image.fromarray(ranimage)

    # set up drawing on image
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), ranface_encodings in zip(ranface_locations, ranface_encodings):

        # compare chosen images with known faces
        matches = face_recognition.compare_faces(known_face_encodings, ranface_encodings)

        name = "Unknown Person"

        if True in matches:
            # get id of known face then return its name
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # draw box
        draw.rectangle(((left - 10, top - 10), (right + 10, bottom + 10)), outline=(227, 236, 75))

        # font size from box size
        diff = (right - left) + (bottom - top)
        font_size = 8 + int(diff/30)

        # draw label
        font = ImageFont.truetype("arial.ttf", font_size)
        text_width = draw.textlength(name, font)

        draw.rectangle(((left - 10, bottom - font_size + 2), (right + 10, bottom + 10)), fill=(227, 236, 75), outline=(227, 236, 75))
        draw.text((left, bottom-font_size), name, (0, 0, 0, 0), font)

    del draw
    pil_image.save(f"identified/{ranface}_scanned.jpg")