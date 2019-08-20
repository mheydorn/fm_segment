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


cap = cv2.VideoCapture(0)

boxes = []

graph_def = tf.GraphDef()
labels = []

# These are set to the default names from exported models, update as needed.
filename = "model1322.pb"

output_layer = 'Softmax:0'
input_node = 'images_input:0'

# Import the TF graph
with tf.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (int(96*5), int(96*5)))
        frame = frame[...,::-1]

        # Display the resulting frame
        softmax, = sess.run(prob_tensor, {input_node: [img] })

        cv2.imshow("whole", frame)
        cv2.imshow("softmax", softmax)
        cv2.waitKey(0)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

























