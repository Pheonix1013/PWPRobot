import cv2
import numpy as np

cap = cv2.VideoCapture("obstacle1.MOV")

# Create the background subtractor object
# history=100 means it learns the floor over 100 frames
backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Refined white range (let's keep it strict now)
lower_white = np.array([0, 0, 180])   
upper_white = np.array([180, 50, 255]) 

kernel = np.ones((5, 5), np.uint8)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print (width, height)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Apply background subtraction to find MOTION
    fgMask = backSub.apply(frame)
    
    # 2. Apply color thresholding to find WHITE
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colorMask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 3. COMBINE THEM: Only keep pixels that are MOVING AND WHITE
    # This ignores the static white reflections on the floor
    combinedMask = cv2.bitwise_and(fgMask, colorMask)

    # 4. Clean up the combined mask
    combinedMask = cv2.morphologyEx(combinedMask, cv2.MORPH_OPEN, kernel, iterations=1)
    combinedMask = cv2.dilate(combinedMask, kernel, iterations=3)

    # 5. Find contours in the combined mask
    cnts, _ = cv2.findContours(combinedMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        # Sort by area and take the largest moving white object
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        if area > 500: # Smaller threshold since we filtered the floor out
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if y>864 and y<1296:
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 4)
                cv2.putText(frame, f"{x},{y}", (int(x) - 40, int(y) - int(radius) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 5)

    # Display windows
    cv2.imshow('Final Tracking', frame)
    cv2.imshow('Motion + Color Mask', combinedMask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
