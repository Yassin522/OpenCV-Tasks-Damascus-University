# Osama Bazo & Yassin Abdulmahdi 
import cv2
import numpy as np

# Global Variables
gaussian_blur_kernel_size = 5
median_blur_kernel_size = 5
canny_threshold_low = 100
canny_threshold_high = 200
erosion_iterations = 1
dilation_iterations = 1
opening_iterations = 4
closing_iterations = 4
image = cv2.imread('5.jpg')
image2 = cv2.imread('output.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary.jpg', binary_image)
binary_image = cv2.imread('binary.jpg')

# Edit Global Variables


def update_gaussian_blur_kernel_size(value):
    global gaussian_blur_kernel_size
    gaussian_blur_kernel_size = max(1, value)
    Gaussian_Median()


def update_median_blur_kernel_size(value):
    global median_blur_kernel_size
    median_blur_kernel_size = max(1, value)
    Gaussian_Median()


def update_canny_threshold_low(value):
    global canny_threshold_low
    canny_threshold_low = value
    Gaussian_Median()


def update_canny_threshold_high(value):
    global canny_threshold_high
    canny_threshold_high = value
    Gaussian_Median()


def update_erosion_iterations(value):
    global erosion_iterations
    erosion_iterations = max(1, value)
    Erosion_and_Dilation()


def update_dilation_iterations(value):
    global dilation_iterations
    dilation_iterations = max(1, value)
    Erosion_and_Dilation()


def update_opening_iterations(value):
    global opening_iterations
    opening_iterations = max(1, value)
    Opening_and_Closing()


def update_closing_iterations(value):
    global closing_iterations
    closing_iterations = max(1, value)
    Opening_and_Closing()

# Morphological Operations

# Gaussian & Median


def Gaussian_Median():
    global image
    global gaussian_blur_kernel_size
    global median_blur_kernel_size
    global canny_threshold_low
    global canny_threshold_high
    gaussian_blur = cv2.GaussianBlur(
        image, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)
    median_blur = cv2.medianBlur(image, median_blur_kernel_size)
    canny_gaussian = cv2.Canny(
        gaussian_blur, canny_threshold_low, canny_threshold_high)
    cv2.imwrite('output.jpg', canny_gaussian)
    cv2.imshow('Canny with Gaussian Blur', canny_gaussian)
    cv2.imshow('Gaussian Blur', gaussian_blur)
    cv2.imshow('Median Blur', median_blur)

# Erosion & Dilation


def Erosion_and_Dilation():
    global image
    global erosion_iterations
    global dilation_iterations
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(binary_image, kernel, iterations=erosion_iterations)
    dilation = cv2.dilate(binary_image, kernel, iterations=dilation_iterations)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Dilation', dilation)
    cv2.imwrite('dilation.jpg', dilation)
    cv2.imwrite('erosion.jpg', erosion)
    
    print("gg")
    print(binary_image.shape)
    print(erosion.shape)
    print("nou")



    logical_and_dilation = cv2.bitwise_and(image, dilation)
    logical_or_dilation = cv2.bitwise_or(image, dilation)
    logical_xor_dilation = cv2.bitwise_xor(image, dilation)
    cv2.imwrite('logical_and_dilation.jpg', logical_and_dilation)
    cv2.imwrite('logical_or_dilation.jpg', logical_or_dilation)
    cv2.imwrite('logical_xor_dilation.jpg', logical_xor_dilation)

    logical_and_erosion = cv2.bitwise_and(image, erosion)
    logical_or_erosion = cv2.bitwise_or(image, erosion)
    logical_xor_erosion = cv2.bitwise_xor(image, erosion)
    cv2.imwrite('logical_and_erosion.jpg', logical_and_erosion)
    cv2.imwrite('logical_or_erosion.jpg', logical_or_erosion)
    cv2.imwrite('logical_xor_erosion.jpg', logical_xor_erosion)



# Opening & Closing


def Opening_and_Closing():
    global image
    global opening_iterations
    global closing_iterations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(
        binary_image, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    closing = cv2.morphologyEx(
        binary_image, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)
    cv2.imwrite('Opening.jpg', opening)
    cv2.imwrite('Closing.jpg', closing)

    logical_and_Opening = cv2.bitwise_and(image, opening)
    logical_or_Opening = cv2.bitwise_or(image, opening)
    logical_xor_Opening = cv2.bitwise_xor(image, opening)
    cv2.imwrite('logical_and_Opening.jpg', logical_and_Opening)
    cv2.imwrite('logical_or_Opening.jpg', logical_or_Opening)
    cv2.imwrite('logical_xor_Opening.jpg', logical_xor_Opening)

    logical_and_closing = cv2.bitwise_and(image, closing)
    logical_or_closing = cv2.bitwise_or(image, closing)
    logical_xor_closing = cv2.bitwise_xor(image, closing)
    cv2.imwrite('logical_and_closing.jpg', logical_and_closing)
    cv2.imwrite('logical_or_closing.jpg', logical_or_closing)
    cv2.imwrite('logical_xor_closing.jpg', logical_xor_closing)


def manual_dilation(binary_image, kernel):
    height, width = binary_image.shape
    k_height, k_width = kernel.shape

    dilated_image = np.zeros((height, width), dtype=np.uint8)

    for x in range(k_height // 2, height - k_height // 2):
        for y in range(k_width // 2, width - k_width // 2):
            if binary_image[x, y] == 255:
                dilated_image[x - k_height // 2:x + k_height //
                              2 + 1, y - k_width // 2:y + k_width // 2 + 1] = 255

    return dilated_image


def manual_erosion(binary_image, kernel):
    height, width = binary_image.shape
    k_height, k_width = kernel.shape

    eroded_image = np.zeros((height, width), dtype=np.uint8)

    for x in range(k_height // 2, height - k_height // 2):
        for y in range(k_width // 2, width - k_width // 2):
            if np.array_equal(binary_image[x - k_height // 2:x + k_height // 2 + 1, y - k_width // 2:y + k_width // 2 + 1], kernel):
                eroded_image[x, y] = 255

    return eroded_image

# Main Function


def main():
    global gaussian_blur_kernel_size
    global median_blur_kernel_size
    global canny_threshold_low
    global canny_threshold_high
    cv2.namedWindow('Original Image')
    cv2.createTrackbar('Gaussian Blur Kernel Size', 'Original Image',
                       gaussian_blur_kernel_size, 15, update_gaussian_blur_kernel_size)
    cv2.createTrackbar('Median Blur Kernel Size', 'Original Image',
                       median_blur_kernel_size, 15, update_median_blur_kernel_size)
    cv2.namedWindow('Canny Thresholds')
    cv2.createTrackbar('Canny Low Threshold', 'Canny Thresholds',
                       canny_threshold_low, 255, update_canny_threshold_low)
    cv2.createTrackbar('Canny High Threshold', 'Canny Thresholds',
                       canny_threshold_high, 255, update_canny_threshold_high)
    Gaussian_Median()
    binary_image = cv2.imread('binary.jpg', 0)

    kernel = np.array([[0, 255, 0],
                       [255, 255, 255],
                       [0, 255, 0]], dtype=np.uint8)

    dilated_image = manual_dilation(binary_image, kernel)

    cv2.imwrite('manually_dilated_image.jpg', dilated_image)

    while True:
        key = cv2.waitKey(1)
        if key == 13:
            break
    cv2.destroyAllWindows()

    # Erosion & Dilation
    global erosion_iterations
    global dilation_iterations
    cv2.namedWindow('Erosion and Dilation Iterations')
    cv2.createTrackbar('Erosion Iterations', 'Erosion and Dilation Iterations',
                       erosion_iterations, 10, update_erosion_iterations)
    cv2.createTrackbar('Dilation Iterations', 'Erosion and Dilation Iterations',
                       dilation_iterations, 10, update_dilation_iterations)
    Erosion_and_Dilation()
    while True:
        key = cv2.waitKey(1)
        if key == 13:
            break

    cv2.destroyAllWindows()

    # Opening & Closing
    global opening_iterations
    global closing_iterations
    cv2.namedWindow('Opening and Closing Iterations')
    cv2.createTrackbar('Opening Iterations', 'Opening and Closing Iterations',
                       opening_iterations, 10, update_opening_iterations)
    cv2.createTrackbar('Closing Iterations', 'Opening and Closing Iterations',
                       closing_iterations, 10, update_closing_iterations)
    Opening_and_Closing()
    while True:
        key = cv2.waitKey(1)
        if key == 13:
            break
    cv2.destroyAllWindows()


main()
