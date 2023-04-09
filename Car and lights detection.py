import numpy as np
import cv2 as cv

cap = cv.VideoCapture(r"C:\Users\Parsa\Documents\PythonProjects\istockphoto-1292798121-640_adpp_is.mp4")

# Create a background subtractor object with detectShadows=False
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

fps = cap.get(cv.CAP_PROP_FPS)
print(fps)

# Define fixed dimensions for light rectangles
light_rect_width = 20
light_rect_height = 20

while(True):
    rec, frame = cap.read()
    if not rec:
        break

    # Apply background subtraction without shadows
    fgmask = fgbg.apply(frame)

    # Perform morphological operations to remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    # Find contours for cars
    contours, _ = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)

        # Check if the contour is a car
        car_threshold = 1000
        if cv.contourArea(contour) > car_threshold:
            color = (150, 0, 0) # Blue color for other moving objects
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Check for red lights inside the car contour
            light_roi = frame[y:y+h,x:x+w]
            hsv_light_roi = cv.cvtColor(light_roi, cv.COLOR_BGR2HSV)

            # Define range of red color in HSV
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])

            # Threshold the HSV image to get only red colors
            mask_red = cv.inRange(hsv_light_roi, lower_red, upper_red)

            # Define range of yellow color in HSV
            lower_yellow = np.array([20,100,100])
            upper_yellow = np.array([30,255,255])

            # Threshold the HSV image to get only yellow colors
            mask_yellow = cv.inRange(hsv_light_roi, lower_yellow, upper_yellow)

            # Combine the masks to obtain all light objects inside the car contour
            mask_light = cv.bitwise_or(mask_red, mask_yellow)

            # Find contours for lights
            contours_light, _ = cv.findContours(mask_light, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for contour_light in contours_light:
                (x_l, y_l, w_l, h_l) = cv.boundingRect(contour_light)

                # Check if the contour is a red light
                red_threshold = 200
                if cv.contourArea(contour_light) > red_threshold and mask_red[y_l:y_l+h_l,x_l:x_l+w_l].any():
                    # Create a rectangle with fixed dimensions at the location of the light
                    x_l += int(w_l/2 - light_rect_width/2)
                    y_l += int(h_l/2 - light_rect_height/2)
                    cv.rectangle(frame, (x+x_l, y+y_l), (x+x_l + light_rect_width, y+y_l + light_rect_height), (0, 0, 255), 2)

                # Check if the contour is a yellow moving light
                yellow_threshold = 200
                if cv.contourArea(contour_light) > yellow_threshold and mask_yellow[y_l:y_l+h_l,x_l:x_l+w_l].any():
                    # Create a rectangle with fixed dimensions at the location of the light
                    x_l += int(w_l/2 - light_rect_width/2)
                    y_l += int(h_l/2 - light_rect_height/2)
                    cv.rectangle(frame, (x+x_l, y+y_l), (x+x_l + light_rect_width, y+y_l + light_rect_height), (0, 255, 255), 2)

    cv.imshow('Cars and moving objects', frame)
    cv.imshow('fgmask', fgmask)

    keyexit = cv.waitKey(30) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()
