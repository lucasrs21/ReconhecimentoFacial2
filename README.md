# Facial recognition with registration phase

The focus of this project is facial recognition to unlock doors or allow access to previously registered people.

## Introduction

The project Facial Recognition with Registration Phase was idealized for frequency control and access permission only for registered people in our IEEE student branch UNESP Bauru. The initial ideia was the implementation of the program into an inteligent lock capable of recognize authorized people to access the room without the need for keys, and better management of who enters the room and how much time they spend there.

## Getting Started

Deployment of the files in the environment, like this:

```
├─ dir
    ├── env
    │   └── (files of the env bin,share...)
    ├── __pycache__
    ├── dataset
    │   ├── person1
    │   │   └── (photos of person1)
    │   └── person2
    │       └── (photos of person2)
    ├── face_detection_model
    │   ├── deploy.prototxt
    │   └── res10_300x300_ssd_iter_140000.caffemodel
    ├── initial_dataset
    ├── model_data
    │   └── deploy.prototxt
    │   └── weights.caffemodel
    ├── new_dataset
    ├── output
    │   ├── embeddings.pickle
    │   ├── le.pickl
    │   └── recognizer.pickle
    ├── updated_dataset
    ├── videos
    │   └── (videos of people to be register)
    ├── Pipfile
    ├── Pipfile.lock
    ├── README.md
    ├── brig_rot_change.py
    ├── cria_dir.py
    ├── machlearn.py
    ├── openface_nn4.small2.v1.t7
    ├── photos_by_video.py
    ├── procimage.py
    ├── procimagelib.py
    └── reconface.py
```

### Prerequisites

What things you need to install the software and how to install them:
* Python3.6 or plus;

* numpy (v1.17.2)
```
pip install numpy
```
* OpenCV(cv2)(v4.1.1.26)
```
pip install opencv-python
pip install opencv-contrib-python
```
* imutils (v0.5.3)
```
pip install imutils
```
* pillow (v6.1.0)
```
pip install pillow
```
* shutil
```
pip install pytest-shutil
```
* sklearn (v0.21.3)
```
pip install scikit-learn
```

## Running
* **Run procimage.py to register a single person**
Note: When running this code if you choose the option 'web' press 'q' to close the webcam window named 'frame' and start the processing of the images obtained.

* **Run machlearn.py to train the neural network**
Note: It's necessary to have 2 different people in the dataset to run properly the face recognition.

* **Run reconface.py to start the face recognition**

## Explanation
The project is divided between 3 main codes: First one is the resgistring code (Procimage.py); second one is the code responsable for tranining the Neural Net to recognize the faces in your dataset (machlearninig.py) and the third one is the code responsable for using the trained net to perform the actual recognition of the registered faces (reconface.py).

The 'procimage.py' is associated to the processing of the image and the obtaining images. It has 2 different ways to capture images for the registration. By webcam live or by video.
First the captured images are saved in the file named "initial_dataset" (where your original images will be). In the second step, the faces of the images are cut to improve precessing and reduce errors, and will be saved in "updated_dataset". In the third and fourth steps, the images are rotated and changed in brightness to diversify the dataset. They are saved in the "new_dataset" and "dataset" folders respectively. The "dataset" folder is the final folder that facial recognition will use to train.

The 'machlearn.py' is the traning phase of the Neural Net and it uses ".pickle" files to store the data referent the labels and face paterns, the generated files are called "le" for labels, "embbedings" for face paterns and "recognizer" for the file created from the merging of the other two.

The 'reconface.py' is the code that uses the trained Neural Net to perform the recognition. It is done through the usage of the ".pickle" file named recognizer, whitch contains all the data referent to the faces on the dataset.

## Vídeo

[![IMAGE ALT TEXT](http://img.youtube.com/vi/3_rxj71geVk/0.jpg)](http://www.youtube.com/watch?v=3_rxj71geVk "Projeto de reconhecimento facial")

## Built With

* [Visual Studio Code](https://code.visualstudio.com/) - IDLE used for development of the project.
* [Caffe Model](https://caffe.berkeleyvision.org/) - Pre-trained neural network model for face detection.

## Authors

* [**Felipe Zechel**](https://github.com/zechelf)
* [**Luanne Barbosa**](https://github.com/Luanne-Barbosa)
* [**Lucas Rodrigues**](https://github.com/lucasr21)
* [**Matheus Augusto**](https://github.com/MatheusMABR)
* [**Mauro Yoshio**](https://github.com/mayokogitgud)
* [**Leonardo Moreno**](https://github.com/leopmoreno) - Management of the team.

## License

This project is free and non-profit. The marketing of it is prohibited.

## Acknowledgments
* Based On: [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/) - [Face detection with OpenCV and deep learning](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)

