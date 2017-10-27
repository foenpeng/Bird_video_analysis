import numpy as np
import cv2
cap = cv2.VideoCapture('c-1C3_6_21_5_30.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

def image_part(img,x,y,w,h):
    part = img[y:y + h, x:x + w]
    return part


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

while(1):
    ret, frame = cap.read()
    frame_part = image_part(frame, 0,80,400,400)
    fgmask = fgbg.apply(frame_part)
    kernel = np.ones((10, 10), np.uint8)
    #thresh_dilate = cv2.dilate(fgmask,kernel,iterations=1)
    thresh_dilate = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
    edges = auto_canny(fgmask)
    cv2.imshow('frame',edges)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
