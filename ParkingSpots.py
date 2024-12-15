import cv2
import pickle
import sys

# Get the video file path from arguments
video_path = sys.argv[1] if len(sys.argv) > 1 else "Try2.mp4"

try:
    with open('CarParkingSpots', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

drawing = False
ix, iy = -1, -1

cap = cv2.VideoCapture(video_path)
success, img = cap.read()
if not success:
    print("Failed to read video")
    exit()

def mouseClick(event, x, y, flags, params):
    global ix, iy, drawing, img, posList

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img2 = img.copy()
            x_min = min(ix, x)
            y_min = min(iy, y)
            x_max = max(ix, x)
            y_max = max(iy, y)
            cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
            for pos in posList:
                x_p, y_p, w_p, h_p = pos
                cv2.rectangle(img2, (x_p, y_p), (x_p + w_p, y_p + h_p), (255, 0, 255), 2)
            cv2.imshow("Image", img2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min = min(ix, x)
        y_min = min(iy, y)
        x_max = max(ix, x)
        y_max = max(iy, y)
        w = x_max - x_min
        h = y_max - y_min
        posList.append((x_min, y_min, w, h))
        with open('CarParkingSpots', 'wb') as f:
            pickle.dump(posList, f)

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1, w, h = pos
            x2, y2 = x1 + w, y1 + h
            if x1 <= x <= x2 and y1 <= y <= y2:
                posList.pop(i)
                with open('CarParkingSpots', 'wb') as f:
                    pickle.dump(posList, f)
                break

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while True:
    img_copy = img.copy()
    for pos in posList:
        x, y, w, h = pos
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow("Image", img_copy)
    cv2.setMouseCallback("Image", mouseClick)
    cv2.resizeWindow("Image", img.shape[1], img.shape[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
