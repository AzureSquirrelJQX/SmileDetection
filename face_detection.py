import argparse
import cv2
import os
import msvcrt

faceCascade = cv2.CascadeClassifier("xmls/haarcascade_frontalface_default.xml")  # Load a pretrained classifier by OpenCV


def detect_face(in_img):
    # return the location of the face detected
    if in_img.ndim == 3:  # Input image in RGB mode
        gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)  # convert the Input image into gray mode. (As required by OpenCV)
    elif in_img.ndim == 1:  # Input image in gray mode
        gray_img = in_img
    face_locations = faceCascade.detectMultiScale(
        gray_img ,  # using default parameters for convenience
        scaleFactor = 1.1,
        minNeighbors = 8,
        minSize = (55, 55),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return face_locations


def main(config):
    try:
        use_camera = config.use_camera  # check if we need to use camera
        if use_camera:
            # real-time detecting face from computer's camera
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try to load the default camera of the PC
            while True:
                retval, frame = cap.read()  # retval: bool indicating whether we could find a camera
                                            # frame: Array containing the picture captured
                if not retval:
                    raise RuntimeError("cannot use the computer's camera")

                face_locations = detect_face(frame)
                for (x, y, w, h) in face_locations:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 255), 2)  # use yellow rectangle to mark the face
                
                cv2.imshow("Real-Time Face Detect", frame)  # show the image in a window
                cv2.waitKey(1)  # refresh the frame per millisecond
                if msvcrt.kbhit() and ord(msvcrt.getch()) == ord('q'):  # press 'q' to terminate the program
                    break

            cap.release()  # release the camera resource
            cv2.destroyAllWindows()  # destroy all the windows
        else:
            # detecting face from input image
            in_image = "{}.jpg".format(config.in_image)
            draw_image = "{}.jpg".format(config.draw_image)
            face_image = config.face_image  # deal with the mutiple images case
            if not os.path.exists(in_image):  # try to load the Input image specified in the parameters
                raise RuntimeError("cannot find {}".format(in_image))

            img = cv2.imread(in_image)  # read the Input image
            face_locations = detect_face(img)
            draw_img = img.copy()  # copy the Input image to the Output file

            detected_faces = []
            for (x, y, w, h) in face_locations:
                cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 225, 255), 2)
                detected_faces.append(img[y:y + h, x:x + w])  # crop the face detected from the original image

            if len(detected_faces) == 1:  # output the face images to the desired locations
                cv2.imwrite("{}.jpg".format(face_image), detected_faces[0])
            else:
                for i in range(len(detected_faces)):
                    cv2.imwrite("{} ({}).jpg".format(face_image, i + 1), detected_faces[i])  # use i + 1 in the filename if there're mutiple faces
            cv2.imwrite(draw_image, draw_img)
    except RuntimeError as e:
        print("Error: {}".format(e.args))
        return 1
    except:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # usage:
    # face_detection.py --in_image example --draw_image example_out --face_image example_face
    # face_detection.py --use_camera True

    parser.add_argument("--in_image", type = str, default = "example", help = "source input image name (without extension .jpg)")
    parser.add_argument("--draw_image", type = str, default = "example_draw", help = "draw face image name (without extension .jpg)")
    parser.add_argument("--face_image", type = str, default = "example_face", help = "face image name (without extension .jpg)")
    parser.add_argument("--use_camera", type = bool, default = False, help = "detect faces from the default camera")

    config = parser.parse_args()

    print(config)  # print out the configuration of the program in terminal
    main(config)
