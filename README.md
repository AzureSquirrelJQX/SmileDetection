# SmileDetection

Smile Detection project codes for course CS124, SJTU. (I am one of the students taking this course. I forked this project from TA Jiaoping Hu.)
* Note: The changes committed by me is licensed under the GNU General Public License v3.0. Remember to state license and copyright notice, state changes, disclose source and use the same license when using this project :) 

# Problem Description

Given a picture of a person, could you tell me whether he/she is smiling? Please let your computer to give the answer.

# Getting Started
## Installation
- This code was tested with Python 3.7, windows 10
- Dataset [GENKI-4K](https://inc.ucsd.edu/mplab/wordpress/wp-content/uploads/genki4k.tar) should be downloaded to train the models. 
- **data_faces** are face images gernerated from orignal GENKI-4K (using opencv face detector).
- **xmls** containes two xml files from opencv.
- **img_label.txt** is the face image names and their labels. The images that cannot be detected faces by opencv are discarded.
- Clone this repo:
```
git clone https://github.com/AzureSquirrelJQX/SmileDetection
cd SmileDetection
```

## Preparing
```
pip3 install numpy
pip3 install opencv-python
pip3 install scikit-learn
pip3 install scikit-image
pip3 install pillow
```

## Task 1: Face Detection with Opencv

- Run ```face_detection.py``` to detect face in example.jpg.
- Run ```face_detection.py --use_camera True``` to detect faces from your camera real-time.

## Task 2: Smile Detection Models Training

- Run ```train_smile_detection_model.py``` to train smile detection models. 10-fold cross validation is utilized.
- Run ```train_smile_detection_model.py --use_hog True``` to use HOG features.
* Note: This program will save the trained SVC models to ```model_x.svc```, and the predict results to ```predicted_x.txt```.

## Task 3: Real-time Smile Detection 

- Run ```realtime_detect_smiles.py --model model_x.svc``` to detect smiles.
- Run ```realtime_detect_smiles.py --model model_x.svc --use_hog True``` to use HOG-based SVC models.
* Note: You have to specify the SVC model explicitly. According to my experiment, if you use LBP features, ```model_6.svc``` has the best predicting result.
