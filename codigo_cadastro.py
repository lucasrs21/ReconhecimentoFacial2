import cv2
from imutils import paths
import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import numpy as np
import os

base_dir = os.path.dirname(__file__)
print(base_dir)

entrada = input('video/web?: ')

if entrada == 'video':
    x = input('Nome do vídeo: ')
    formato_video = input('Formato do vídeo: ')
    # set video file path of input video with name and extension
    vid = cv2.VideoCapture(base_dir + '/videos/' + x + '.' + formato_video)

if entrada == 'web':
    x = input('Nome da pessoa a ser cadastrada: ')
    # turn on the webcam
    vid = cv2.VideoCapture(0)

fps = 10

pasta = base_dir + '/dataset/' + x
if not os.path.exists(pasta):
    os.makedirs(pasta)

#for frame identity
index = 0
aux = 1
while(True):
    # Extract images
    ret, frame = vid.read()
    
    if entrada == 'web':  
        #show the output frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # keyboard 'q' closes/stop
        if key == ord('q'):
            break
    
    # end of frames
    if not ret: 
        break
    # Saves images
    if index%fps == 0:
        index = aux
        name = pasta + '/frame' + str(index) + '.jpg'      
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        aux += 1
    # next frame
    index += 1
#---------------------------------------------------------------------------
# Extract_Embeddings--------------------------------------------------------

# pegando os arquivos necessários ao programa
basedir = os.path.dirname(__file__)
output = basedir + "/output"

dataset = basedir + "/dataset"
embeddings = output + "/embeddings.pickle"
embedding_model = basedir + "/openface_nn4.small2.v1.t7"
face_model = basedir + "/face_detection_model"


# load our serialized face detector from disk
protoPath = face_model + "/deploy.prototxt"
modelPath = face_model + "/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > 0.9:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddings, "wb")
f.write(pickle.dumps(data))
f.close()
#---------------------------------------------------------------------------
#Train_Model----------------------------------------------------------------

# pegando os arquivos necessários ao programa
# basedir = os.path.dirname(__file__)
# output = basedir + "/output"

# embeddings = output + "/embeddings.pickle"
label = output + "/le.pickle"
recog = output + "/recognizer.pickle"


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(embeddings, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(recog, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(label, "wb")
f.write(pickle.dumps(le))
f.close()