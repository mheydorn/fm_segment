import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import re
import glob
import scipy.misc
import os
import pickle
from IPython import embed
import cv2
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import datetime
import time
import os
import random
import glob

import cv2
import numpy as np
from IPython import embed


cap = cv2.VideoCapture(2)


NUM_ROWS = 5
NUM_COLS = 5

boxes = []



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
  
    # Display the resulting frame
    cv2.imshow("whole", frame)
    stepSizeRow = int(frame.shape[0] / NUM_ROWS) -1
    stepSizeCol = int(frame.shape[1] / NUM_COLS) -1

    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            img = frame[stepSizeRow * r : stepSizeRow * (r+1) , stepSizeCol * c  : stepSizeCol * (c+1)]
            cv2.imshow("part", img)
            cv2.waitKey(0)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

























