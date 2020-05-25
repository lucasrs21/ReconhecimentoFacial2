# Facial recognition with registration phase

The focus of this project is facial recognition to unlock doors or allow access to previously registered people.

## Introduction

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

* numpy;
```
pip install numpy
```
* OpenCV(cv2);
```
pip install opencv-python
pip install opencv-contrib-python
```
* imutils;
```
pip install imultils
```
* pillow;
```
pip install pillow
```
* shutils;
```
pip install pytest-shutil
```
* sklearn;
```
pip install scikit-learn
```
## Executing
Run procimage.py to register a single person
Run reconface.py to start the recognition face

note: It's necessary to have 2 different people in the dataset to run properly the face recognition.

## Explanition
The procimage.py is associated to the processing of the image. It has 2 different ways to capture images for the registration. By webcam live or by video.
First the captured images are saved in the file named "initial_dataset" (where your original images will be). In the second step, the faces of the images are cut to improve precessing and reduce errors, and will be saved in "updated_dataset". In the third and fourth steps, the images are rotated and changed in brightness to diversify the dataset. They are saved in the "new_dataset" and "dataset" folders respectively. The "dataset" folder is the final folder that facial recognition will use to train.

## Built With

* [Visual Studio Code](https://code.visualstudio.com/) - IDLE used for development of the project.
* [Caffe Model](https://caffe.berkeleyvision.org/) - Pre-trained neural network model for face detection.

## Authors

* [**Felipe Zechel**](https://github.com/zechelf)
* [**Luanne Barbosa**]()
* [**Lucas Rodrigues**]()
* [**Matheus Augusto**](https://github.com/MatheusMABR)
* [**Mauro Yoshio**]()

## License

This project is free and non-profit. The marketing of it is prohibited.

## Acknowledgments
* Based On: [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/) - [Face detection with OpenCV and deep learning](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)


