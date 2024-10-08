# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run face landmarker."""

import argparse
import time
from datetime import datetime
import numpy as np

import cv2
import logging
import mediapipe as mp
from kivy.clock import Clock
from kivy.core.window import Window

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import kivy

kivy.require('2.1.0')  # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as UixImage
from kivy.graphics.texture import Texture

from PIL import Image

import os

print(os.uname())

if os.uname()[0] == 'Darwin':
    print('Darwin detected (MacOS)')
else:
    print('sysname:', os.uname()[0], 'nodename:', os.uname()[1])
    # sysname=Linux / nodename=raspberrypi -> Raspberry Pi
    from picamera2 import Picamera2

from media_pipe_utils import get_ear_values, draw_landmarks_on_image

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None
BUSY = False
LEFT_BLINK_COUNTER, RIGHT_BLINK_COUNTER = 0, 0
LEFT_OPEN_COUNTER, RIGHT_OPEN_COUNTER = 0, 0

#ear_left_label = None

cam = None  # Camera in desktops


class LoginScreen(GridLayout):

    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        # self.login_screen = None
        # global ear_left_label
        self.cols = 3

        col1 = GridLayout(cols=1)
        col2 = GridLayout(cols=1)
        col3 = GridLayout(cols=1)

        self.ear_left_label = Label(text='EAR L:')
        col1.add_widget(self.ear_left_label)
        self.ear_right_label = Label(text='EAR R:')
        col3.add_widget(self.ear_right_label)

        self.left_blinks = Label(text='0')
        col1.add_widget(self.left_blinks)
        self.right_blinks = Label(text='0')
        col3.add_widget(self.right_blinks)

        self.left_opens = Label(text='0')
        col1.add_widget(self.left_opens)
        self.right_opens = Label(text='0')
        col3.add_widget(self.right_opens)

        self.fps = Label(text='FPS')
        col1.add_widget(self.fps)
        self.filler = Label(text='Singleton Sistemas')
        col3.add_widget(self.filler)

        texture = Texture.create(size=(100, 100), colorfmt="rgb")
        arr = np.ndarray(shape=[100, 100, 3], dtype=np.uint8)
        # fill your numpy array here
        arr.fill(255)  # or img[:] = 255
        data = arr.tostring()
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        self.image = UixImage(texture=texture)

        #self.image = UixImage(source='Teste.jpg')
        col2.add_widget(self.image)

        self.add_widget(col1)
        self.add_widget(col2)
        self.add_widget(col3)


class MyApp(App):

    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        global cam

        if os.uname()[0] == 'Linux' and os.uname()[1] == 'raspberrypi':
            # Inicializa a câmera no Raspberry Pi
            self.picam2 = Picamera2()
            self.picam2.start()
        else:
            cam = cv2.VideoCapture(0)
            cv2.namedWindow("test")

        self.fps_avg_frame_count = 10

        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            result_callback=self.save_result)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        fh = logging.FileHandler(datetime.now().strftime('blinklogfile_%Y_%m_%d_%H_%M_%s.log'))
        fh.setLevel(logging.DEBUG)  # ensure all messages are logged to file
        # create a formatter and set the formatter for the handler.
        frmt = logging.Formatter("%(asctime)s$%(message)s")
        fh.setFormatter(frmt)
        # add the Handler to the logger
        self.logger.addHandler(fh)

    def on_start(self):
        #Window.custom_titlebar = True
        Window.maximize()

    def build(self):
        self.login_screen = LoginScreen()
        #p1 = Process(target=self.main())
        #p1.start()
        #p1.join()
        #main()
        #x = threading.Thread(target=self.main(), daemon=True)
        #x.start()
        Clock.schedule_interval(self.do_detection, 1 / 30)
        return self.login_screen

    def save_result(self, result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % self.fps_avg_frame_count == 0:
            FPS = self.fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    def do_detection(self, dt):
        global BUSY, cam
        # Verifica se está fazendo uma deteccao no momento
        if BUSY:
            return
        BUSY = True

        im_array = None
        if os.uname()[0] == 'Linux' and os.uname()[1] == 'raspberrypi':
            im_array = self.picam2.capture_array("main")
        else:
            ret, im_array = cam.read()
            cv2.imshow("test", im_array)

        # image = Image.fromarray(im_array)

        image = cv2.cvtColor(im_array, cv2.COLOR_RGBA2BGR)
        #print(image)

        im = Image.fromarray(im_array)
        #print('PIL Image:', im)

        # 0 -> flip horizontally, 1 -> flip vertically
        # image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face landmarker using the model.
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        #print(fps_text)

        # DETECTION_RESULT é setado no callback...
        # print(DETECTION_RESULT)

        if DETECTION_RESULT is not None:
            #print(DETECTION_RESULT, len(DETECTION_RESULT.face_landmarks))
            if len(DETECTION_RESULT.face_landmarks) > 0:
                #print('EAR:', get_ear_values(DETECTION_RESULT))
                #print(self.login_screen.ear_left_label)
                #print(DETECTION_RESULT.face_blendshapes[0])
                face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in
                                          DETECTION_RESULT.face_blendshapes[0]]
                face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in
                                           DETECTION_RESULT.face_blendshapes[0]]
                eye_blink_left = face_blendshapes_scores[face_blendshapes_names.index('eyeBlinkLeft')]
                eye_blink_right = face_blendshapes_scores[face_blendshapes_names.index('eyeBlinkRight')]

                if self.login_screen.ear_left_label is not None:
                    global COUNTER, LEFT_BLINK_COUNTER, RIGHT_BLINK_COUNTER
                    global LEFT_OPEN_COUNTER, RIGHT_OPEN_COUNTER
                    ear_left = get_ear_values(DETECTION_RESULT)[0]
                    ear_right = get_ear_values(DETECTION_RESULT)[1]
                    self.logger.info(
                        "{:.3f}${:.3f}${:.3f}${:.3f}".format(ear_left, ear_right, eye_blink_left, eye_blink_right,
                                                             COUNTER))
                    #if ear_left < 0.35:
                    if eye_blink_left > 0.4:
                        LEFT_BLINK_COUNTER += 1
                    else:
                        LEFT_OPEN_COUNTER += 1
                    #if ear_right < 0.35:
                    if eye_blink_right > 0.4:
                        RIGHT_BLINK_COUNTER += 1
                    else:
                        RIGHT_OPEN_COUNTER += 1
                    self.login_screen.ear_left_label.text = 'EAR ESQ = {:.3f}\n{:.3f}'.format(ear_left, eye_blink_left)
                    self.login_screen.ear_right_label.text = 'EAR DIR = {:.3f}\n{:.3f}'.format(ear_right,
                                                                                               eye_blink_right)
                    self.login_screen.left_blinks.text = str(LEFT_BLINK_COUNTER)
                    self.login_screen.right_blinks.text = str(RIGHT_BLINK_COUNTER)
                    self.login_screen.left_opens.text = str(LEFT_OPEN_COUNTER)
                    self.login_screen.right_opens.text = str(RIGHT_OPEN_COUNTER)
                    self.login_screen.fps.text = fps_text

            arr = np.ndarray(shape=[400, 400, 3], dtype=np.uint8)
            arr.fill(0)  # or img[:] = 255
            mp_blank_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

            annotated_image = draw_landmarks_on_image(
                #self.login_screen.image.export_as_image(),
                #Image.fromarray(self.login_screen.image),
                mp_blank_image.numpy_view(),
                DETECTION_RESULT
            )

            vert_flip_image = np.array(list(reversed(annotated_image)))
            #print(annotated_image)
            texture = Texture.create(size=(400, 400), colorfmt="rgb")
            data = vert_flip_image.tostring()
            texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
            self.login_screen.image.texture = texture

        BUSY = False

    def start(self, model: str, num_faces: int,
              min_face_detection_confidence: float,
              min_face_presence_confidence: float, min_tracking_confidence: float,
              camera_id: int, width: int, height: int) -> None:
        """Continuously run inference on images acquired from the camera.

        Args:
              model: Name of the face landmarker model bundle.
              num_faces: Max number of faces that can be detected by the landmarker.
              min_face_detection_confidence: The minimum confidence score for face
                detection to be considered successful.
              min_face_presence_confidence: The minimum confidence score of face
                presence score in the face landmark detection.
              min_tracking_confidence: The minimum confidence score for the face
                tracking to be considered successful.
              camera_id: The camera id to be passed to OpenCV.
              width: The width of the frame captured from the camera.
              height: The height of the frame captured from the camera.
        """

        # global ear_left_label

        # Start capturing video input from the camera
        '''cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)'''

        picam2 = Picamera2()
        picam2.start()

        # Visualization parameters
        row_size = 50  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 0)  # black
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        # Label box parameters
        label_background_color = (255, 255, 255)  # White
        label_padding_width = 1500  # pixels

        def save_result(result: vision.FaceLandmarkerResult,
                        unused_output_image: mp.Image, timestamp_ms: int):
            global FPS, COUNTER, START_TIME, DETECTION_RESULT

            # Calculate the FPS
            if COUNTER % fps_avg_frame_count == 0:
                FPS = fps_avg_frame_count / (time.time() - START_TIME)
                START_TIME = time.time()

            DETECTION_RESULT = result
            COUNTER += 1

        # Initialize the face landmarker model
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            result_callback=save_result)
        detector = vision.FaceLandmarker.create_from_options(options)

        # Continuously capture images from the camera and run inference
        #while cap.isOpened():
        while True:
            '''success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )'''

            im_array = picam2.capture_array("main")
            # image = Image.fromarray(im_array)

            image = cv2.cvtColor(im_array, cv2.COLOR_RGBA2BGR)
            #print(image)

            #cv2.imshow('face_landmarker', image)
            #key = cv2.waitKey(0)

            im = Image.fromarray(im_array)
            #print('PIL Image:', im)

            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Run face landmarker using the model.
            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(FPS)
            #print(fps_text)
            '''text_location = (left_margin, row_size)
            current_frame = image
            cv2.putText(current_frame, fps_text, text_location,
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size, text_color, font_thickness, cv2.LINE_AA)'''

            # DETECTION_RESULT é setado no callback...
            #print(DETECTION_RESULT)

            if DETECTION_RESULT is not None:
                #print('EAR:', get_ear_values(DETECTION_RESULT))
                #print(self.login_screen.ear_left_label)
                if self.login_screen.ear_left_label is not None:
                    global COUNTER
                    self.login_screen.ear_left_label.text = str(get_ear_values(DETECTION_RESULT)[0])
                    self.login_screen.ear_right_label.text = str(get_ear_values(DETECTION_RESULT)[1])
                    #self.login_screen.ear_left_label.text = str(COUNTER)
                    self.login_screen.canvas.ask_update()
                    time.sleep(1)

            #cv2.imshow('face_landmarker', current_frame)

            # Stop the program if the ESC key is pressed.
            if cv2.waitKey(1) == 27:
                break

        detector.close()
        #cap.release()
        #cv2.destroyAllWindows()

    def main(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--model',
            help='Name of face landmarker model.',
            required=False,
            default='face_landmarker.task')
        parser.add_argument(
            '--numFaces',
            help='Max number of faces that can be detected by the landmarker.',
            required=False,
            default=1)
        parser.add_argument(
            '--minFaceDetectionConfidence',
            help='The minimum confidence score for face detection to be considered '
                 'successful.',
            required=False,
            default=0.5)
        parser.add_argument(
            '--minFacePresenceConfidence',
            help='The minimum confidence score of face presence score in the face '
                 'landmark detection.',
            required=False,
            default=0.5)
        parser.add_argument(
            '--minTrackingConfidence',
            help='The minimum confidence score for the face tracking to be '
                 'considered successful.',
            required=False,
            default=0.5)
        # Finding the camera ID can be very reliant on platform-dependent methods.
        # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS,
        # starting from 0.
        # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
        # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
        parser.add_argument(
            '--cameraId', help='Id of camera.', required=False, default=0)
        parser.add_argument(
            '--frameWidth',
            help='Width of frame to capture from camera.',
            required=False,
            default=1280)
        parser.add_argument(
            '--frameHeight',
            help='Height of frame to capture from camera.',
            required=False,
            default=960)
        args = parser.parse_args()

        self.start(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
                   args.minFacePresenceConfidence, args.minTrackingConfidence,
                   int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    #main()
    #p1 = Process(target=main)
    #p1.start()
    MyApp().run()
