import cv2
import time
from threading import Thread
import time
import cv2
import numpy as np
from multiprocessing import Pool
import os
import concurrent.futures
import time
from playsound import playsound

if os.name == "posix":
    import tflite_runtime.interpreter as tflite
elif os.name == "nt":
    import tensorflow as tf


class WebcamStream:
    """
    Faster Real-Time Video Processing using Multi-Threading 
    """

    def __init__(self, src=0):
        self.src = src   # default is 0 for primary cameraq

        # opening video capture stream
        self.cap = cv2.VideoCapture(self.src)
        if src == 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if self.cap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.cap.get(5))
        print(f"FPS of webcam hardware/input stream: {fps_input_stream}")

        # reading a single frame from cap stream for initializing
        self.grabbed, self.frame = self.cap.read()
        #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.cap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        # daemon threads keep running in the background while the program is executing
        self.t.daemon = True

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

    # method for reading next frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.cap.read()
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.cap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True


class TFLite:
    """
    TensorFlow Lite lets you run machine-learned models with low latency
    """

    input_mean = 127.5  # input_mean, input_std
    input_std = 127.5  # input_mean, input_std
    labels = ["0 P", "1 N"]  # 0 FIRE, 1 BACKGROUND
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
        self.img = cv2.resize(image, (224, 224))
        # add N dim
        self.input_data = np.expand_dims(self.img, axis=0)
        self.input_data = (np.float32(self.input_data) -
                           self.input_mean) / self.input_std
        self.interpreter.set_tensor(
            self.input_details[0]["index"], self.input_data)
        self.interpreter.invoke()
        self.output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"])
        self.results = np.squeeze(self.output_data)
        self.pred = self.results.argsort()[-5:][::-1]
        self.pred = int(self.pred[0])
        return self.pred


class BackgroundSubtractor:
    """
    Background Subtraction is used for generating a foreground mask
    """

    def __init__(self):
        # set parameters for BackgroundSubtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40)  # history=100

    def subtract_background(self, frame):
        # update the background model
        self.fg_mask = self.bg_subtractor.apply(frame)
        _, self.thresh = cv2.threshold(
            self.fg_mask, 100, 250, cv2.THRESH_BINARY)

        self.contours, self.hier = cv2.findContours(
            self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return self.contours, self.hier


def count_fire(detected_list):

    if len(detected_list) > 20:
        detected_list.pop(0)  # Remove the first element
        if detected_list.count(0) > 3:
            playsound('warning.mp3')


if __name__ == "__main__":
    # initializing and starting multi-threaded webcam capture input stream
    # src = 0 is for primary camera
    # src = 'vlc-record.mp4'
    src = '8min.mp4'

    webcam_stream = WebcamStream(src=src)
    webcam_stream.start()

    # Create tflite object for prediction
    tflite = TFLite()

    # Create BackgroundSubtractor object for movement detection
    subtractor = BackgroundSubtractor()

    # Create a list to count fire occurances
    detected_list = []

    # processing frames in input stream
    num_frames_processed = 0
    start = time.time()

    while True:
        if webcam_stream.stopped is True:
            break
        else:
            frame = webcam_stream.read()

            ######### detection & prediction & alarm ############
            contours, hier = subtractor.subtract_background(frame)
            for detected in contours:
                # Set the valid detection size
                size = cv2.contourArea(detected)

                # size condition filters out noise and shaky frame
                if (size > 100) and (size < 100000):
                    x, y, w, h = cv2.boundingRect(detected)
                    height, width = frame.shape[:2]

                    xmin, ymin = (x - int(w/2), y - int(h/2))
                    xmax, ymax = (x + w), (y + h)

                    try:
                        # get ROI (region of interest)
                        ROI = frame[ymin:ymax, xmin:xmax]
                        if src == 0:
                            ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                        cv2.imshow("ROI", ROI)
                        pred = tflite.read_image(ROI)

                        # save detected image files
                        filename = str(time.strftime("%Y%m%d-%H%M%S")) + ".jpg"

                        if pred == 0:
                            print("FIRE")
                            cv2.imwrite(f"0.FIRE.{filename}", ROI)
                            detected_list.append(0)  # Add the last element
                        else:
                            print("BACKGROUND")
                            cv2.imwrite(f"1.BACKGROUND.{filename}", ROI)
                            detected_list.append(1)

                    except:
                        print(f"ROI: {ROI}")

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(count_fire, detected_list)

        # adding a delay for simulating time taken for processing a frame
        delay = 0.03  # delay value in seconds. so, delay=1 is equivalent to 1 second
        time.sleep(delay)
        num_frames_processed += 1

        # cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()
    webcam_stream.stop()  # stop the webcam stream

    # printing time elapsed and fps
    elapsed = end-start
    fps = num_frames_processed/elapsed
    print(f"FPS: {fps} , Elapsed Time: {elapsed} , Frames Processed: {num_frames_processed}")

    # closing all windows
    cv2.destroyAllWindows()
