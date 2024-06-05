from facedb import FaceDB
from PIL import Image

db = FaceDB(path="facedata")

def register_face(webcam_image, user_name):
    pil_image = Image.fromarray(webcam_image)
    img_path = f"{user_name}.jpg"
    pil_image.save(img_path)
    db.add(user_name, img=img_path)
    return webcam_image, "Face registered successfully."

def query_face(webcam_image):
    result = db.recognize(img=webcam_image)
    if result:
        return webcam_image, f"Recognized as {result['name']} with confindence {result['confidence']}"
    else:
        return webcam_image, "Unknown face"
