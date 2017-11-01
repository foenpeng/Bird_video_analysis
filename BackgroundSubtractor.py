import numpy as np
import cv2
cap = cv2.VideoCapture('c-1C3_6_21_5_30.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2, backgroundRatio=0.5)

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

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

def contour_method(img):
    dilated = cv2.dilate(img,None,iterations=1)
    _,cnts,_ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    largest = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
    cv2.drawContours(img, largest, -1, (255,255,0), )


while(1):
    blank = np.zeros((400, 400, 1), dtype=np.uint8)
    ret, frame = cap.read()
    frame_part = image_part(frame, 0,80,350,400)
    edges = auto_canny(frame_part)
    cv2.circle(edges, (300, 170), 150, (0, 0, 0), 3)
    fgmask = fgbg.apply(edges)
    skel = skeletonize(fgmask)

    kernel = np.ones((3, 3), np.uint8)
    thresh_dilate = cv2.dilate(fgmask,kernel,iterations=3)
    opening = cv2.morphologyEx(thresh_dilate, cv2.MORPH_CLOSE, kernel)

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)
    kp, des = surf.detectAndCompute(fgmask, None)
    #print("# kps: {}, descriptors: {}".format(len(kp), des.shape))
    img = cv2.drawKeypoints(fgmask, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    cv2.imshow('frame',frame)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
