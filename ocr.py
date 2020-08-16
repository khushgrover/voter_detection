import cv2 
import pytesseract

for i in range(4):
    img = cv2.imread('r'+str(i)+'.jpg')
    #img = cv2.GaussianBlur(img,(5,5),0)
    # img = cv2.medianBlur(img,5) 
    #retval, img = cv2.threshold(img,150,255, cv2.THRESH_BINARY)
    cv2.imshow('img',img)
    cv2.waitKey(3000)
    custom_config = r'--oem 3 --psm 6'
    txt = pytesseract.image_to_string(img, lang='eng', config=custom_config)
    print(txt)