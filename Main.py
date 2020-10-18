# Author Mohammad Akbarzadeh
# https://github.com/mohammadakz

# Importing nested detection class
import os
from nested_detection import Nested_Detection


# Model and video file names and path
CWD_PATH = os.getcwd()

model_name = 'inference_graph_50000_balance'
video_name = 'mk2.mp4'
out_put = 'out_put/results_{}.avi'.format(video_name[:-4])
label_path = 'label_map_person.pbtxt'

# Performing the detection
dt = Nested_Detection()
dt.worker_detection(model_name, label_path, video_name,out_put)
