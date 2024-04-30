from picamera2 import Picamera2
import time
picam2 = Picamera2()
picam2.start()
time.sleep(1)
image = picam2.capture_image("main")
print(image)