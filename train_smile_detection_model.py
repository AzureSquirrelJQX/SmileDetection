import argparse
import os
import numpy as np
from sklearn.model_selection import KFold
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import cv2
import pickle


def load_image(image_name):
    in_img = cv2.imread(image_name)
    if not os.path.exists(image_name):  # try to load the Input image specified in the parameters
        raise RuntimeError("cannot find {}".format(image_name))

    if in_img.ndim == 3:  # Input image in RGB mode
        return cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)  # convert the Input image into gray mode. (As required by OpenCV)
    elif in_img.ndim == 1:  # Input image in gray mode
        return in_img[::]


def lbp(image):
    # get lbp feature
    # This function is originally provided by Mao & Yang & Zhao, 
    # modified and commented by Junqi Xie
    # input image: one gray face image, np.array
    # output hist: lbp feature vecter, np.array

    # divide the image into (block * block) sections to deal with images of different resolutions
    block = 7  # the number of blocks on each edge
    width = image.shape[1]  # width of image
    height = image.shape[0]  # height of image
    column = width // block  # the number of pixels on a column
    row = height // block  # the number of pixels on a row

    hist = np.array([])
    for i in range(block * block):
        lbp1 = local_binary_pattern(image[row * (i // block):row * ((i // block) + 1), \
            column * (i % block):column * ((i % block) + 1)], 8, 1, "default")  # get the lbp feature of a block
        hist1, _ = np.histogram(lbp1, density = True, bins = 256, range = (0, 256))  # calculate the histogram of the feature
        hist = np.concatenate((hist, hist1))  # concatenate the histogram with the previous ones
    return hist


def hog_feature(image):
    # get hog feature
    # input image: one gray face image, np.array
    # output hist: hog feature vecter, np.array

    # divide the image into (block * block) sections to deal with images of different resolutions
    block = 7  # the number of blocks on each edge
    width = image.shape[1]
    height = image.shape[0]
    column = width // block  # the number of pixels on a column
    row = height // block  # the number of pixels on a row

    hist = np.array([])
    for i in range(block * block):
        hist1 = hog(image[row * (i // block):row * ((i // block) + 1), column * (i % block):column * ((i % block) + 1)], \
            pixels_per_cell = (column, row), cells_per_block = (1, 1))  # get the hog feature of a block (histogram)
        hist = np.concatenate((hist, hist1))  # concatenate the histogram with the previous ones
    return hist


def main(config):
    try:
        # step 1: get image names and labels
        print("loading image names and labels...")
        img_label = "{}.txt".format(config.img_label)
        if not os.path.exists(img_label):  # try to load the file specified in the parameters
            raise RuntimeError("cannot find {}".format(img_label))

        img_names = []
        labels = []
        with open(img_label, "r") as f:  # read labels and convert it into np.array
            lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            img_names.append(data[0])
            labels.append(int(data[1]))
        labels = np.array(labels)


        # step 2: get features
        print("computing image features... need a few minutes, please wait...")
        faces_path = config.faces_path
        if not os.path.exists("{}\\".format(faces_path)):  # try to load the folder specified in the parameters
            raise RuntimeError("cannot find {}\\".format(faces_path))

        if config.use_hog == False:
            # use lbp to get features
            features = lbp(load_image("{}\{}".format(faces_path, img_names[0])))
            for i in range(1, len(img_names)):  # stack feature arrays in sequence vertically
                features = np.vstack((features, lbp(load_image("{}\{}".format(faces_path, img_names[i])))))
        else:
            # use hog to get features
            features = hog_feature(load_image("{}\{}".format(faces_path, img_names[0])))
            for i in range(1, len(img_names)):  # stack feature arrays in sequence vertically
                features = np.vstack((features, hog_feature(load_image("{}\{}".format(faces_path, img_names[i])))))


        # step 3: training and testing, using 10-fold cross validation
        print("10-fold cross validation...")
        kf = KFold(n_splits = 10, random_state = 2019, shuffle = True)  # K-Folds cross-validator
        kf.get_n_splits(lines)
        for i, (train_index, test_index) in enumerate(kf.split(lines)):
            features_train, features_test = features[train_index], features[test_index]  # get features
            labels_train, labels_test = labels[train_index], labels[test_index]  # get labels

            svc = SVC(kernel = "linear", degree = 2, gamma = 1, coef0 = 0)  # use SVM to classify
            svc.fit(features_train, labels_train)  # trining data
            with open("model_{}.svc".format(i), "wb") as fout:
                pickle.dump(svc, fout)  # store the svc model to reuse later
            predict_result = svc.predict(features_test)  # use testing data to get result
            
            f1 = f1_score(labels_test, predict_result)
            acc = accuracy_score(labels_test, predict_result)

            with open("predicted_{}.txt".format(i), "w") as fout:
                predict_cnt = 0
                for j in test_index:
                    fout.write("{} {} {}\n".format(img_names[j], labels[j], predict_result[predict_cnt]))
                    predict_cnt += 1

            print("fold {}, f1: {:.5f}, acc: {:.5f}, save svc model in file model_{}.svc, \
predicted results in file predicted_{}.txt".format(i, f1, acc, i, i))
    except RuntimeError as e:
        print("Error: {}".format(e.args))
        return 1
    except:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # usage:
    # train_smile_detection_model.py --faces_path data_faces --img_label img_label
    # face_detection.py --use_hog True

    parser.add_argument("--faces_path", type = str, default = "data_faces", help = "source input images path")
    parser.add_argument("--img_label", type = str, default = "img_label", help = "text file containing labels of the images")
    parser.add_argument("--use_hog", type = bool, default = False, help = "use hog feature instead of lbp")

    config = parser.parse_args()

    print(config)  # print out the configuration of the program in terminal
    main(config)
