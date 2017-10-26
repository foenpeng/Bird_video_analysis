import cv2
import numpy as np

def define_ROI():
    cap = cv2.VideoCapture('c-1C3_6_21_5_30.avi')
    ret, frame = cap.read()
    c, r, w, h = 0, 175, 150, 150
    cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)
    cv2.imshow('defien_ROI',frame)
    cv2.waitKey(0)


def run_main():
    cap = cv2.VideoCapture('c-1C3_6_21_5_30.avi')
    # Read the first frame of the video
    ret, frame = cap.read()
    # Set the ROI (Region of Interest). Actually, this is a
    # rectangle of the building that we're tracking
    c, r, w, h = 0, 175, 150, 150
    track_window = (c, r, w, h)
    # Create mask and normalized histogram
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((50., 130., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)
    cv2.imshow('ROI', mask)
    #cv2.imwrite('ref.jpg',frame)
    cv2.waitKey(5000)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.putText(frame, 'Tracked', (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #define_ROI()
    run_main()