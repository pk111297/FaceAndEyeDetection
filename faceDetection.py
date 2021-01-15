import cv2
import numpy 
import sys
#taking the umage which we want to face detect
pictureName=sys.argv[1]
#opening the given imageName
picture=cv2.imread(pictureName,cv2.IMREAD_UNCHANGED)
#converting the  image in grayScale as for face detection image should be in grayscale mode
grayScalePicture=cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
#taking the sample data for detecting face
faceDetectors=cv2.CascadeClassifier('XMLs/haarcascade_frontalface_default.xml')
#Now detecting the faces by calling the method and it returns list of tuples which has top left  coordinates and height and width of rectangle
faceCoordinates=faceDetectors.detectMultiScale(grayScalePicture,scaleFactor=1.3,minNeighbors=5)
#looping over the faceCoordinates array
for face in faceCoordinates:
	x,y,width,height=face
	#drawing the red rectangle on original image 
	cv2.rectangle(picture,(x,y),(x+width,y+height),color=(0,0,255),thickness=2)
cv2.imshow("Image",picture)
cv2.waitKey(0)

