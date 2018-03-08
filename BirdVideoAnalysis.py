import numpy as np
import cv2

cap = cv2.VideoCapture('c-1C3_6_21_5_30.avi')

sensor_position = (289,122)
flower_top_line_p1 = (305, 102)
flower_top_line_p2 = (280, 150)

def cnt_fitelps(img,cnt):
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    return ellipse

def cnt_fitline(img,cnt):
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

def image_part(img,x,y,w,h):
    part = img[y:y + h, x:x + w]
    return part

def clean_circle(img):
    # function to clean the circle drawn in every frame
    blank = np.zeros((480, 640, 1), dtype=np.uint8)
    cv2.circle(blank, (300, 250), 150, (255, 255, 255), 2)
    dst = cv2.inpaint(img, blank, 3, cv2.INPAINT_TELEA)
    return dst

i = 0
while(cap.isOpened()):
    #print(i)
    i+=1

    ret, frame = cap.read()
    if not ret:
        break

    # pre-processing image and find bird contour
    frame = clean_circle(frame)
    frame_part = image_part(frame, 0,80,350,400)
    blw = cv2.cvtColor(frame_part,cv2.COLOR_RGB2GRAY)
    ret, thr_blw = cv2.threshold(blw,80,255,cv2.THRESH_BINARY_INV)
    _, cnts, _ = cv2.findContours(thr_blw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blank = np.zeros(thr_blw.shape, dtype="uint8")

    # check bird position
    if cnts:
        largest = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        bird_img = cv2.drawContours(blank.copy(), [largest], -1, 255, -1)

        # find the centroid
        M = cv2.moments(largest)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # normal area size should be between 3000 and 15000
        area = cv2.contourArea(largest)

        # check whether the bird's body intersect with flower top plane
        line = cv2.line(blank.copy(), flower_top_line_p1, flower_top_line_p2, 255, 3)
        intersect = cv2.bitwise_and(bird_img, line)
        intersect_area = np.count_nonzero(intersect)

        # check the distance from the sensor to the closest bird contour pixel
        distance = cv2.pointPolygonTest(largest, sensor_position, measureDist=True)

        # fit the bird with an ellipse to estimate the orientation
        ellipse = cnt_fitelps(blw,largest)

        cv2.drawContours(blw, [largest], -1, 255, -1)
        cv2.line(blw, flower_top_line_p1, flower_top_line_p2, 255, 3)
        cv2.circle(blw,(289,122),2,255,-1)
        cv2.putText(blw,"Distance:{0}    Intersect:{1}".format(round(distance,1), intersect_area),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        cv2.imshow("orig", blw )
        k = cv2.waitKey(1000) & 0xff
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()
