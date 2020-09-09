# Author Mohammad Akbarzadeh

# Import packages
import os
import sys
import cv2
import math
from shapely.geometry import Polygon
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import tensorflow as tf

sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


class Nested_detection():

    def __init__(self):
        self.NUM_CLASSES = 1
        self.small_worker = 20
        self.mid_worker = 35
        self.large_worker = 55
        self.n_small = 25
        self.n_mid = 25
        self.n_large = 25
        self.skip_frames = 30
        self.font = cv2.FONT_HERSHEY_PLAIN

    def sub_regions(self, img, width, height):

        num_small = int((width / self.small_worker) / self.n_small) + 1
        self.small_width = int((width / num_small))

        num_mid = int((width / self.mid_worker) / self.n_mid) + 1
        self.mid_width = int((width / num_mid))

        num_large = int((width / self.large_worker) / self.n_large) + 1
        self.large_width = int((width / num_large))

        # (x,y) # (0,0) till (1920, 540)
        up_left_small = (0, 0)
        down_right_small = (width, int((height / 2)))

        # (0,270) till (1920, 810)
        up_left_mid = (0, int((height / 4)))
        down_right_mid = (width, height - int(height / 4))

        # (0,540) till (1920, 540)
        up_left_large = (0, int((height / 2)))
        down_right_large = (width, height)

        # (y:y, x:x)
        main_small_region = img[up_left_small[1]:down_right_small[1], up_left_small[0]:down_right_small[0]]
        main_mid_region = img[up_left_mid[1]:down_right_mid[1], up_left_mid[0]:down_right_mid[0]]
        main_large_region = img[up_left_large[1]:down_right_large[1], up_left_large[0]:down_right_large[0]]

        small_sub_regions = []
        small_sub_regions_overlap = []

        mid_sub_regions = []
        mid_sub_regions_overlap = []

        large_sub_regions = []
        large_sub_region_overlap = []

        for i in range(0, width, self.small_width):
            small = main_small_region[:, i:i + self.small_width]
            small_sub_regions.append(small)

        for i in range(int(self.small_width / 2), (width - int(self.small_width / 2)), self.small_width):
            small_overlap = main_small_region[:, i:i + self.small_width]
            small_sub_regions_overlap.append(small_overlap)

        for i in range(0, width, self.mid_width):
            mid = main_mid_region[:, i:i + self.mid_width]
            mid_sub_regions.append(mid)

        for i in range(int(self.mid_width / 2), (width - int(self.mid_width / 2)), self.mid_width):
            mid_overlap = main_mid_region[:, i:i + self.mid_width]
            mid_sub_regions_overlap.append(mid_overlap)

        for i in range(0, width, self.large_width):
            large = main_large_region[:, i:i + self.large_width]
            large_sub_regions.append(large)

        for i in range(int(self.large_width / 2), (width - int(self.large_width / 2)), self.large_width):
            large_overlap = main_large_region[:, i:i + self.large_width]
            large_sub_region_overlap.append(large_overlap)

        return small_sub_regions, small_sub_regions_overlap, mid_sub_regions, \
               mid_sub_regions_overlap, large_sub_regions, large_sub_region_overlap

    def read_video(self, video_name, out_put):
        video = cv2.VideoCapture(video_name)
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        out = cv2.VideoWriter(out_put, cv2.VideoWriter_fourcc(*'XVID'),
                              30, (frame_width, frame_height))
        self.frame_index = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret == True:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.frame_index % self.skip_frames == 0:
                    s_regions, s_regions_overlap, m_regions, m_regions_overlap, \
                    l_regions, l_regions_overlap = self.sub_regions(frame_rgb, frame_width, frame_height)
                    self.frame_index += 1
                    return s_regions, s_regions_overlap, m_regions, m_regions_overlap, \
                           l_regions, l_regions_overlap

                else:
                    self.frame_index += 1
                    out.write(frame)

    def worker_detection(self, model_name, label_path, video_name, out_put):

        path_to_check = os.path.join(model_name, 'frozen_inference_graph.pb')
        categories = label_map_util.convert_label_map_to_categories(label_map_util.load_labelmap(label_path),
                                                                    max_num_classes=1,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_check, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        s_regions, s_regions_overlap, m_regions, m_regions_overlap, \
        l_regions, l_regions_overlap = self.read_video(video_name, out_put)

        detection_results = []

        for idx, subregion in enumerate(s_regions):
            frame_expanded_1 = np.expand_dims(subregion, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded_1})

            coordinates_1 = vis_util.return_coordinates(
                subregion,
                self.frame_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.50)

            for coordinate in coordinates_1:
                (ymin, ymax, xmin, xmax, acc, classification) = coordinate
                detection_results.append(
                    [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax, int(acc),
                     classification])

        for idx, subregion in enumerate(s_regions_overlap):
            frame_expanded_1 = np.expand_dims(subregion, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded_1})

            coordinates_1 = vis_util.return_coordinates(
                subregion,
                self.frame_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.50)

            for coordinate in coordinates_1:
                (ymin, ymax, xmin, xmax, acc, classification) = coordinate
                detection_results.append(
                    [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax, int(acc),
                     classification])

        for idx, subregion in enumerate(m_regions):
            frame_expanded_1 = np.expand_dims(subregion, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded_1})

            coordinates_1 = vis_util.return_coordinates(
                subregion,
                self.frame_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.50)

            for coordinate in coordinates_1:
                (ymin, ymax, xmin, xmax, acc, classification) = coordinate
                detection_results.append(
                    [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax, int(acc),
                     classification])

        for idx, subregion in enumerate(m_regions_overlap):
            frame_expanded_1 = np.expand_dims(subregion, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded_1})

            coordinates_1 = vis_util.return_coordinates(
                subregion,
                self.frame_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.50)

            for coordinate in coordinates_1:
                (ymin, ymax, xmin, xmax, acc, classification) = coordinate
                detection_results.append(
                    [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax, int(acc),
                     classification])

        for idx, subregion in enumerate(l_regions):
            frame_expanded_1 = np.expand_dims(subregion, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded_1})

            coordinates_1 = vis_util.return_coordinates(
                subregion,
                self.frame_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.50)

            for coordinate in coordinates_1:
                (ymin, ymax, xmin, xmax, acc, classification) = coordinate
                detection_results.append(
                    [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax, int(acc),
                     classification])

        for idx, subregion in enumerate(l_regions_overlap):
            frame_expanded_1 = np.expand_dims(subregion, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded_1})

            coordinates_1 = vis_util.return_coordinates(
                subregion,
                self.frame_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.50)

            for coordinate in coordinates_1:
                (ymin, ymax, xmin, xmax, acc, classification) = coordinate
                detection_results.append(
                    [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax, int(acc),
                     classification])