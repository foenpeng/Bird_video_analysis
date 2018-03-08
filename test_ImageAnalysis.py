import numpy as np
import cv2

def image_part(img,x,y,w,h):
    part = img[y:y + h, x:x + w]
    return part

def draw_circle(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print("something")
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
        print(x,y)

def clean_circle(img):
    # function to clean the circle drawn in every frame
    blank = np.zeros((480, 640, 1), dtype=np.uint8)
    cv2.circle(blank, (300, 250), 150, (255, 255, 255), 2)
    dst = cv2.inpaint(img, blank, 3, cv2.INPAINT_TELEA)
    return dst

img = cv2.imread("6_origin.jpg")
frame = clean_circle(img)
frame_part = image_part(frame, 0, 80, 350, 400)
blw = cv2.cvtColor(frame_part, cv2.COLOR_RGB2GRAY)
ret, thr_blw = cv2.threshold(blw, 80, 255, cv2.THRESH_BINARY_INV)
_, cnts, _ = cv2.findContours(thr_blw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
blank = np.zeros(thr_blw.shape, dtype="uint8")
if cnts:
    # c = max(cnts, key=cv2.contourArea)
    largest = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    bird_img = cv2.drawContours(blank.copy(), [largest], -1, 255, -1)
    distance = cv2.pointPolygonTest(largest,(289,122),measureDist=True)
    line = cv2.line(blank.copy(), (305,102),(280,150),255,3)
    intersect = cv2.bitwise_and(bird_img, line)
    print(distance, np.sum(intersect))
    cv2.imshow("intersect",intersect)
    cv2.waitKey(5000)

cv2.circle(blw,(289,122),2,255,-1)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
#bbox = cv2.selectROI("img",blw)
#p1 = (int(bbox[0]), int(bbox[1]))
#p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#cv2.rectangle(blw, p1, p2, (255, 0, 0), 2, 1)
#print(bbox)
# out.write(blank)

while(1):
    cv2.imshow('image',blw)
    if cv2.waitKey(5000) & 0xFF == 27:
        break
cv2.destroyAllWindows()