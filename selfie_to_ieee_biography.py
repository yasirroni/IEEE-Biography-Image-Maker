import cv2
import os

def selfie_to_ieee_biography(img_path, save=False, save_path=None, cascade_path=None):
    # Read the input image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Get img shape
    h_img, w_img = img.shape

    # Get face
    x, y, w, h = get_face(img, cascade_path=cascade_path)
    # print(w, h, w/h)

    # Expand face
    x_, y_, w_, h_ = expand_face_area(x, y, w, h, w_img, h_img)

    # Resize whole img
    r = 300 / w_ # or r = 375 / h
    resize_dim = round(w_img * r), round(h_img * r)
    img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)

    # Map x, y to resized img
    x__ = round(x_ * r)
    y__ = round(y_ * r)

    # Crop
    img = img[y__:y__+375,x__:x__+300]
    
    # Save cropped image
    if save:
        if save_path is None:
            basename = os.path.basename(img_path).split('.')[0]
            dirname = os.path.dirname(img_path)
            save_path = os.path.join(dirname, basename + '_IEEE.png')
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

def expand_face_area(x, y, w, h, w_img, h_img):
    # Shift x, y
    x_ = max(0, x - 0.25 * w)
    y_ = max(0, y - 0.425 * h)

    # Expand w, h 
    w_ = w * 1.5
    h_ = h * 1.875
    w__ = min(w_img - x_, w_)
    h__ = min(h_img - y_, h_)

    bound_ratio = min(w__/w_, h__/h_)

    return x_, y_, w * 1.5 * bound_ratio, h * 1.875 * bound_ratio

def show_img(img):
    from matplotlib import pyplot as plt
    
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()
