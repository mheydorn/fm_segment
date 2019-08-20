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

graph_def = tf.GraphDef()
labels = []

# These are set to the default names from exported models, update as needed.
filename = "model1221.pb"

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
        frame = cv2.resize(frame, (int(96*5), int(96*5)))
        predictionMap = np.zeros((frame.shape[0], frame.shape[1]))


        # Display the resulting frame
        cv2.imshow("whole", frame)
        stepSizeRow = int(frame.shape[0] / NUM_ROWS) 
        stepSizeCol = int(frame.shape[1] / NUM_COLS) 

        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                img = frame[stepSizeRow * r : stepSizeRow * (r+1) , stepSizeCol * c  : stepSizeCol * (c+1)]

                img = cv2.resize(img, (int(96), int(96)))


                predictions, = sess.run(prob_tensor, {input_node: [img] })

                predictionMap[stepSizeRow * r : stepSizeRow * (r+1) , stepSizeCol * c  : stepSizeCol * (c+1)] = predictions[:,:,0]

                '''
                cv2.imshow("part", img)
                cv2.imshow("predictions", predictions[:,:,0])

                '''
        cv2.imshow("stitched predictions", predictionMap)
        cv2.waitKey(1)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

























