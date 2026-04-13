import cv2
import numpy as np

def nothing(x):
    pass

def calibrate_color_lab(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img = cv2.resize(img, (512, 512))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 400, 300)

    cv2.createTrackbar('L Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('L Max', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('a Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('a Max', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('b Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('b Max', 'Trackbars', 255, 255, nothing)

    print("--- L*a*b* Calibration Tool ---")
    print("Press 'ESC' to exit and print the final values.")

    while True:
        l_min = cv2.getTrackbarPos('L Min', 'Trackbars')
        l_max = cv2.getTrackbarPos('L Max', 'Trackbars')
        a_min = cv2.getTrackbarPos('a Min', 'Trackbars')
        a_max = cv2.getTrackbarPos('a Max', 'Trackbars')
        b_min = cv2.getTrackbarPos('b Min', 'Trackbars')
        b_max = cv2.getTrackbarPos('b Max', 'Trackbars')

        lower_bound = np.array([l_min, a_min, b_min])
        upper_bound = np.array([l_max, a_max, b_max])

        mask = cv2.inRange(lab, lower_bound, upper_bound)
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Original Image', img)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\n--- Final L*a*b* Bounds ---")
            print(f"Lower Bound: [{l_min}, {a_min}, {b_min}]")
            print(f"Upper Bound: [{l_max}, {a_max}, {b_max}]")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_color_lab("data/test/good/006.png")