import cv2
import numpy as np
import os
import glob

## TODO: DELETE!!!!


def crop(image_path) -> np.array:
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour and its bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


def crop_linedrawings(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image since the background is white and the object is black
    inverted_gray = cv2.bitwise_not(gray)

    # Apply threshold
    _, thresh = cv2.threshold(inverted_gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize min and max coordinates with the first contour
    if contours:
        min_x, min_y = image.shape[1], image.shape[0]
        max_x = max_y = 0

        # Go through all contours and find the min and max extent
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            min_x, max_x = min(min_x, x), max(max_x, x + w)
            min_y, max_y = min(min_y, y), max(max_y, y + h)

        # Crop the image to the bounding box of all contours
        cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image


def process_images_in_folder(folder_path):
    for image_file in glob.glob(folder_path + "/**/*.png", recursive=True):
        print(image_file)
        # Crop the image
        cropped_image = crop(image_file)

        # Save the cropped image
        cv2.imwrite(image_file, cropped_image)


# Example usage
folder_path = "assets/enns_rensink_1991/png"
process_images_in_folder(folder_path)
