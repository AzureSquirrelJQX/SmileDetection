import argparse
import cv2
import os
import msvcrt
import pickle
import numpy as np
import face_detection
import train_smile_detection_model


def main(config):
    try:
        # real-time detecting smiles from computer's camera
        with open(config.model, "rb") as fin:
            svc = pickle.load(fin)  # load svc mode from the file saved

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try to load the default camera of the PC
        while True:
            retval, frame = cap.read()  # retval: bool indicating whether we could find a camera
                                        # frame: Array containing the picture captured
            if not retval:
                raise RuntimeError("cannot use the computer's camera")

            draw_img = frame.copy()
            face_locations = face_detection.detect_face(frame)
            for (x, y, w, h) in face_locations:
                cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 225, 255), 2)  # use yellow rectangle to mark the face
                face = frame[y:y + h, x:x + w]  # crop the face from the frame
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # convert face image to gray mode

                if config.use_hog == False:
                    feature = train_smile_detection_model.lbp(gray_face)  # get lbp feature
                else:
                    feature = train_smile_detection_model.hog_feature(gray_face)  # get hog feature
                predict_result = svc.predict(np.array([feature]))  # use model to predict smiling

                if predict_result == 1:
                    cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # use red rectangle to mark the smiling face
                    cv2.putText(draw_img, "Smiling", (x, y - 7), 3, 1.2, (0, 255, 0), 1, cv2.LINE_AA)  # add a "Smiling" label

            cv2.imshow("Real-Time Smile Detect", draw_img)  # show the image in a window
            cv2.waitKey(1)  # refresh the frame per millisecond
            if msvcrt.kbhit() and ord(msvcrt.getch()) == ord('q'):  # press 'q' to terminate the program
                break

        cap.release()  # release the camera resource
        cv2.destroyAllWindows()  # destroy all the windows
    except RuntimeError as e:
        print("Error: {}".format(e.args))
        return 1
    except:
        print("Error: unknown error")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # usage:
    # realtime_detect_smiles.py --model model_0.svc --use_hog False

    parser.add_argument("--model", type = str, help = "the file containing the classifying model")
    parser.add_argument("--use_hog", type = bool, default = False, help = "use hog feature instead of lbp")

    config = parser.parse_args()

    print(config)  # print out the configuration of the program in terminal
    main(config)
