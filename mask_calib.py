import cv2
import numpy as np

def nothing(x):
    """Dummy function for trackbar callbacks."""
    pass

def calibrate_color(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Resize for easier viewing on standard monitors (512x512)
    # We do this so the windows don't overlap and hide the trackbars
    img = cv2.resize(img, (512, 512))
    
    # 3. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. Create a window for the Trackbars
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 400, 300)

    # IMPORTANT: In OpenCV, Hue ranges from 0 to 179 (not 360)
    # Saturation and Value range from 0 to 255 (not 100%)
    cv2.createTrackbar('H Min', 'Trackbars', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'Trackbars', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Trackbars', 255, 255, nothing)

    print("--- HSV Calibration Tool ---")
    print("Adjust the sliders until ONLY your target object is WHITE in the Mask window.")
    print("Press 'ESC' to exit and print the final values.")

    while True:
        # Read the current positions of all trackbars
        h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
        h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
        s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
        s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
        v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
        v_max = cv2.getTrackbarPos('V Max', 'Trackbars')

        # Create the lower and upper bounds
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Generate the binary mask based on the bounds
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply the mask to the original image to see the colored result
        result = cv2.bitwise_and(img, img, mask=mask)

        # Show the windows
        cv2.imshow('Original Image', img)
        cv2.imshow('Mask (White = Target)', mask)
        cv2.imshow('Result (Filtered Color)', result)

        # Wait for the ESC key to be pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\n--- Final HSV Bounds ---")
            print(f"Lower Bound: [{h_min}, {s_min}, {v_min}]")
            print(f"Upper Bound: [{h_max}, {s_max}, {v_max}]")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace this with the path to one of your real photos
    calibrate_color("data/test/good/000.png")