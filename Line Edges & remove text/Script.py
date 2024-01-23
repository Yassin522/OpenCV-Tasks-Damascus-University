from __future__ import print_function
import sys
import math
import cv2 as cv
import numpy as np

from common import Sketcher
 
INPUT_IMAGE = "jerry.jpg"
IMAGE_NAME = INPUT_IMAGE[:INPUT_IMAGE.index(".")]
OUTPUT_IMAGE = IMAGE_NAME + "_output.jpg"
TABLE_IMAGE = IMAGE_NAME + "_mask.jpg"


def main(argv):

    try:
        fn = sys.argv[1]
    except:
        fn = INPUT_IMAGE


    image2 = cv.imread(cv.samples.findFile(fn))
 
    if image2 is None:
        print('Failed to load image file:', fn)
        sys.exit(1)
 
    image_mark = image2.copy()
    sketch = Sketcher('Image', [image_mark], lambda : ((255, 255, 255), 255))

    print("press r please to make a mask :)")
 
    while True:
        ch = cv.waitKey()
        if ch == 27: 
            break
        if ch == ord('r'): 
            break
        if ch == ord(' '): 
            image_mark[:] = image2
            sketch.show()
 
    lower_white = np.array([0,0,255])
    upper_white = np.array([255,255,255])
 
    mask2 = cv.inRange(image_mark, lower_white, upper_white)
    cv.imshow('mask', mask2)
    cv.imwrite(TABLE_IMAGE, mask2)


    
    default_file = 'lines.jpeg'
    filename = argv[0] if len(argv) > 0 else default_file
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    
    dst2 = cv.Canny(src, 50, 200, None, 3)
    cdst = cv.cvtColor(dst2, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv.HoughLines(dst2, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
    linesP = cv.HoughLinesP(dst2, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)



    image = cv.imread("jerry.jpg")
    mask = cv.imread("jerry_mask.jpg", 0)

    dst =cv.inpaint(image,mask,3,cv.INPAINT_TELEA)


    cv.imshow("OriginalImage",image)
    cv.imshow('mask2',mask)
    cv.imshow('inpaint',dst)
    cv.imwrite("Result.jpg", dst)


######################################################################################



    image = cv.imread("cattext3.jpg")

    # convert image to gray scale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # threshhold the image and convert to binary image
    ret, binary_image = cv.threshold(gray_image, thresh=245, maxval=255, type=cv.THRESH_BINARY)

    # remove noise
    kernel=np.ones((5,5),np.uint8)
    mask = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)

    # inpaint the image
    dst =cv.inpaint(image,mask,inpaintRadius=5,flags=cv.INPAINT_TELEA)

    # show results
    cv.imshow("OriginalImage",image)
    cv.imshow("Grey image", gray_image)
    cv.imshow("threshholded", binary_image)
    cv.imshow('mask',mask)
    cv.imshow('dst',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.waitKey(0)

    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])