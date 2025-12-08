to run the api on the web version:
python3 -m uvicorn newapi:app --host 0.0.0.0 --port 5000 --reload


ip address for robot: 192.168.240.23

file size is 1280x960


crop test:

import cv2

cap = cv2.VideoCapture(0)

# define the crop box (x, y, width, height)
x, y, w, h = 320,240,640,480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- CROP ---
    cropped = frame[y:y+h, x:x+w]

    # --- PROCESS CROP (example: convert to gray) ---
    processed = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # --- DRAW RECTANGLE ON ORIGINAL IMAGE ---
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)

    # --- DISPLAY ---
    cv2.imshow("Full Frame", frame)
    cv2.imshow("Processed Crop", processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


