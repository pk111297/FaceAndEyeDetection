import cv2
import numpy 
import sys
#taking the umage which we want to face detect
pictureName=sys.argv[1]
#opening the given imageName
picture=cv2.imread(pictureName,cv2.IMREAD_UNCHANGED)
#converting the  image in grayScale as for face detection image should be in grayscale mode
grayScalePicture=cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
#taking the sample data for detecting face,eye and eye with glasses
faceDetectors=cv2.CascadeClassifier('XMLs/haarcascade_frontalface_default.xml')
eyeDetectors=cv2.CascadeClassifier('XMLs/haarcascade_eye.xml')
eyeDetectorsWithGlasses=cv2.CascadeClassifier('XMLs/haarcascade_eye_tree_eyeglasses.xml')
#Now detecting the faces by calling the method and it returns list of tuples which has top left  coordinates and height and width of rectangle
faceCoordinates=faceDetectors.detectMultiScale(grayScalePicture,scaleFactor=1.3,minNeighbors=5)
#looping over the faceCoordinates array
for face in faceCoordinates:
	x,y,width,height=face
	#drawing the red rectangle on original image 
	cv2.rectangle(picture,(x,y),(x+width,y+height),color=(0,0,255),thickness=2)
	#Now as face is detected and so in that each face rectangle will detect the eye
	#Now taking rectangle out from image
	recGrayPicture=grayScalePicture[y:y+height,x:x+width]
	recPicture=picture[y:y+height,x:x+width]
	#Now taking the normal eye haar cascade
	eyeCoordinates=eyeDetectors.detectMultiScale(recGrayPicture)
	for eye in eyeCoordinates:
		#drawing green rectangle over eye
		eyeX,eyeY,eyeWidth,eyeHeight=eye
		cv2.rectangle(recPicture,(eyeX,eyeY),(eyeX+eyeWidth,eyeY+eyeHeight),color=(0,255,0),thickness=2)
	#if eye is not detected from normal eyeDetector then we are using the classifier with the glasses
	#print(len(eyeCoordinates),eyeCoordinates)
	if len(eyeCoordinates)<1:
		eyeCoordinates=eyeDetectorsWithGlasses.detectMultiScale(recGrayPicture)
		#print(len(eyeCoordinates),eyeCoordinates)
		for eye in eyeCoordinates:
			eyeX,eyeY,eyeWidth,eyeHeight=eye
			cv2.rectangle(recPicture,(eyeX,eyeY),(eyeX+eyeWidth,eyeY+eyeHeight),color=(0,255,0),thickness=2)
cv2.imshow("Image",picture)
cv2.waitKey(0)
imageSaveName=(pictureName[0:pictureName.index('.')]+"Detected"+pictureName[pictureName.index('.'):])
cv2.imwrite(imageSaveName,picture)

