import cv2 as cv
import numpy as np
import os

def save_roi(img, roi, caption = "0", out_path = "output"):
    '''Save a particular region of interes of img as a new image.

    Optionally specify end of filename in caption.
    '''
    x0 = roi[0,0]
    y0 = roi[0,1]
    x1 = x0 + roi[0,2]
    y1 = y0 + roi[0,3]

    new_im = img[y0:y1,x0:x1,:]
    if(not os.path.exists(out_path)): os.mkdir(out_path)
    cv.imwrite(os.path.join(out_path, "face"+caption+".jpg"), new_im)
    

def main():
    img = cv.imread("testim.jpg")
    roi = np.array([[414, 293, 897, 897]]) # [x, y, w, h]
    print(img.shape)
    save_roi(img, roi)

if __name__ == "__main__":
    main()
