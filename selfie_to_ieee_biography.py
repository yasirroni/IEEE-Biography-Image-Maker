import cv2
import os

def selfie_to_ieee_biography(img_path, save_path=None, cascade_path=None):
    # Read the input image
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    # Get face
    x, y, w, h = get_face(img, cascade_path=cascade_path)
    
    # Expand face area
    xmin, ymin, xmax, ymax = expand_face_area(x, y, w, h)
    
    # Crop and resize
    img = crop_and_resize(img, xmin, ymin, xmax, ymax)
    
    # Save cropped image
    if save_path:
        cv2.imwrite(save_path, img)
        
    return img
    
def get_face(img, cascade_path=None):
    # Load the cascade
    if cascade_path is None:
        cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.1, 10)

    # Get x, y, w, h
    x, y, w, h = faces[0]
    
    return x, y, w, h

def expand_face_area(x, y, w, h):
    # Expand face area
    # TODO: no need to use xmid, ymid
    xmid = x + w / 2
    ymid = y + h / 2

    xmin = int(xmid - 0.75 * w)
    xmax = int(xmid + 0.75 * w)
    ymin = int(ymid - 0.875 * h)
    ymax = int(ymid + 1 * h)
    
    return xmin, ymin, xmax, ymax

def crop_and_resize(img, xmin, ymin, xmax, ymax, resize_dim=(300, 375)):
    # crop and resize
    img = cv2.resize(img[ymin:ymax,xmin:xmax], resize_dim, interpolation=cv2.INTER_AREA)    
    
    return img

def show_img(img):
    from matplotlib import pyplot as plt
    
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()
