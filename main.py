#This code uses openCV to initialize the webcam and capture images
#images are then compared with a trained CNN to identify a letter
#from the sign alphabet

#Sources: loicmarie
#edited by Lu Bilyk, Ferencz Dominguez, Nik Kershaw, Peter Socha

#Libraries
import numpy as np
import cv2
<<<<<<< HEAD
=======
from image_proc import define_Picture
>>>>>>> 0dbe6c3553fdd1d057cd15675ff0c0301b268d38
import tensorflow as tf
import sys
import os

from pkg_resources import parse_version


OPCV3 = parse_version(cv2.__version__) >= parse_version('3')

# Config GPU as processing system
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def capPropId(prop):
  return getattr(cv2 if OPCV3 else cv2.cv,
    ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)


# Start camera
capture = cv2.VideoCapture(0)

# Set window height and width
capture.set(capPropId("FRAME_WIDTH"), 640)
capture.set(capPropId("FRAME_HEIGHT"), 480)


# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Receive frame captured with the sign symbol and make a prediction
# on letter based on trained CNN
# Returns prediction and its accuracy percentage
def nnetwork(image):

    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    max = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            max = human_string

    return max, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                in tf.gfile.GFile("Output_Labels/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("Output_Graph/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#Start TensoFlow session
with tf.Session() as sess:

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    i = 0

    interpret, accuracy = '', 0.0

    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()

        if ret == True:

            frame = cv2.flip(frame, 1)

            # Create a content area to scan hand gesture
            x1, y1, x2, y2 = 100, 100, 300, 300
            cap_cropped = frame[y1:y2, x1:x2]

            i += 1

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            standard = cv2.resize(gray, (200, 200))

            image_data = cv2.imencode('.jpg', cap_cropped)[1].tostring()

            if i == 9:
                # pass tbe image file to neural network to be tested against
                interpret, accuracy = nnetwork(image_data)

            # Display Letter
            cv2.putText(frame, '%s' % (interpret.upper()), (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 4)

            # Display Accuracy
            cv2.putText(frame, "Accuracy: %0.3f" % (accuracy*100) + '%', (200, 450),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

            # Display detection area
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("img", frame)
            img_sequence = np.zeros((200, 1200, 3), np.uint8)

            # Quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Simple delay
            i += 1
            if i == 20:
                i = 0
            continue

        else:
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
