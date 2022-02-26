import time
import cv2
import numpy as np
from multiprocessing import Pool
import os
import sys
import argparse


if os.name == "posix":
    # import tflite_runtime.interpreter as tflite
    from image_classifier import ImageClassifier
    from image_classifier import ImageClassifierOptions

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Name of image classification model.",
        required=False,
        default="model.tflite",
    )

    # Initialize the image classification model
    options = ImageClassifierOptions(num_threads=4)
    classifier = ImageClassifier(options)

    # LED pins
    B = 13
    G = 19
    R = 26
    second = 0.4

    """  LED blink """
    from gpiozero import LED
    from time import sleep

    def blink(pin):
        try:
            led = LED(pin)
            led.on()
            sleep(second)
            led.off()
        except:
            led.off()


elif os.name == "nt":
    import tensorflow as tf

    # import tflite_runtime.interpreter as tflite


class TFLite:
    """
    TensorFlow Lite lets you run machine-learned models with low latency
    """

    input_mean = 127.5  # input_mean, input_std
    input_std = 127.5  # input_mean, input_std
    labels = ["0 FIRE", "1 BACKGROUND", "2 HUMAN"]
    width, height = 224, 224

    def __init__(self):
        if os.name == "posix":
            self.interpreter = tflite.Interpreter(model_path="model.tflite")
        if os.name == "nt":
            self.interpreter = tf.lite.Interpreter(model_path="model.tflite")

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def read_image(self, image):
        img = cv2.resize(image, (224, 224))
        # add N dim
        input_data = np.expand_dims(img, axis=0)
        input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        results = np.squeeze(output_data)
        top_k = results.argsort()[-5:][::-1]
        top_k = int(top_k[0])
        # return results, top_k
        return top_k


class BackgroundSubtractorKNN:
    """
    Background Subtraction is used for generating a foreground mask
    """

    def __init__(self):
        # set parameters for BackgroundSubtractorKNN
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN()
        self.bg_subtractor.setHistory(20)  # 20
        # remove image sensor noise
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def subtract_background(self, frame):
        # update the background model
        fg_mask = self.bg_subtractor.apply(frame)
        ret, thresh = cv2.threshold(fg_mask, 100, 250, cv2.THRESH_BINARY)

        # Removing image sensor noise
        cv2.erode(thresh, self.erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, self.dilate_kernel, thresh, iterations=2)

        contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours, hier


def count_fire(detected_list, fire):
    """ count the number of fire detections 
    """
    return detected_list.count(fire)


if __name__ == "__main__":

    # Create tflite object for prediction
    tflite = TFLite()

    # Create BackgroundSubtractorKNN object for movement detection
    subtractor = BackgroundSubtractorKNN()

    # Create a list to count fire occurances
    detected_list = []

    # capture input
    file = "fire.mp4"
    # file = 'Fireplace.mp4'
    cap = cv2.VideoCapture(file)
    success, frame = cap.read()

    while success:
        contours, hier = subtractor.subtract_background(frame)
        for detected in contours:
            # Set the valid detection size
            size = cv2.contourArea(detected)

            if size > 800:
                x, y, w, h = cv2.boundingRect(detected)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                # ROI = frame[y - h : y + h, x -w : x + w]

                # get resolution and coordinates
                height, width = frame.shape[:2]

                xmin, ymin = (x - w), (y - h)
                xmax, ymax = (x + w), (y + h)

                # fetch roi
                if (xmin >= 0) and (ymin >= 0) and (xmax < width) and (ymax < height):
                    ROI = frame[ymin:ymax, xmin:xmax]
                    print(f"roi")
                else:
                    ROI = frame[
                        0 : int(height / 2), int(width * 1 / 4) : int(width * 2 / 4)
                    ]  # if the ROI is located in the corner, evaluate the center area
                    print(f"frame")

                # get ROI (region of interest)
                cv2.imshow("ROI", ROI)
                # cv2.imshow("Input", frame)

                # make a prediction
                if os.name == "posix":
                    args = parser.parse_args()
                    top_k = int(classifier.classify(ROI))

                elif os.name == "nt":
                    top_k = tflite.read_image(ROI)

                # save detected image files
                filename = str(time.strftime("%Y%m%d-%H%M%S")) + ".jpg"

                if top_k == 0:
                    print("FIRE")
                    # cv2.imwrite(f"0.FIRE.{filename}", ROI)
                    detected_list.append(0)  # Add the last element
                #                     if os.name == "posix":
                #                         with Pool(5) as p:
                #                             p.map(blink, [R])  # Red
                elif top_k == 1:
                    print("BACKGROUND")
                    # cv2.imwrite(f'1.BACKGROUND.{filename}', ROI)
                    detected_list.append(1)
                #                     if os.name == "posix":
                #                         with Pool(5) as p:
                #                             p.map(blink, [G])  # Green

                elif top_k == 2:
                    print("HUMAN")
                    # cv2.imwrite(f"2.HUMAN.{filename}", ROI)
                    detected_list.append(2)
                #                     if os.name == "posix":
                #                         with Pool(5) as p:
                #                             p.map(blink, [B])  # Blue

                else:
                    print("an error has occured")

                """ Count fire occurances (fire == 0) and set alarm  
                """
                if len(detected_list) > 10:
                    detected_list.pop(0)  # Remove the first element
                    #                     print(f"fire in list: {count_fire(detected_list, 0)} times")
                    if count_fire(detected_list, 0) > 3:
                        print(f"fire in list: {count_fire(detected_list, 0)} times")
                        # if os.name == "posix":
                        #     with Pool(5) as p:
                        #         p.map(blink, [R])  # Red
                else:
                    pass

        keyboard = cv2.waitKey(30)
        if keyboard == "q" or keyboard == 27:
            break
        success, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
