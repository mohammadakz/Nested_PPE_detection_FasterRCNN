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
import json

sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


class Nested_detection():

    def __init__(self):
        self.NUM_WORKER_CLASSES = 1
        self.frame_index = 0
        self.small_worker = 20
        self.mid_worker = 35
        self.large_worker = 55
        self.n_small = 25
        self.n_mid = 25
        self.n_large = 25
        self.skip_frames = 30
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.rectangle_bgr = (255, 255, 255)
        self.detection_results = []
        self.track_list = []
        self.ct = CentroidTracker()
        self.util_match_table = {}
        self.util_detection_results = {}

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
        return small_sub_regions, small_sub_regions_overlap, mid_sub_regions, mid_sub_regions_overlap, \
               large_sub_regions, large_sub_region_overlap

    def calculate_iou(self, box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou

    def remove_duplicates(self, detected_workers):
        duplications = []
        i = 0
        while i <= len(detected_workers):
            j = i + 1
            while j < len(detected_workers):
                iou = self.calculate_iou(
                    [[detected_workers[i][0], detected_workers[i][2]], [detected_workers[i][1], detected_workers[i][2]],
                     [detected_workers[i][1], detected_workers[i][3]],
                     [detected_workers[i][0], detected_workers[i][3]]],
                    [[detected_workers[j][0], detected_workers[j][2]], [detected_workers[j][1], detected_workers[j][2]],
                     [detected_workers[j][1], detected_workers[j][3]],
                     [detected_workers[j][0], detected_workers[j][3]]])

                if iou > 0.50:
                    # or abs(detected_workers[i][0] - detected_workers[j][0]) <= 40 and abs(
                    # detected_workers[i][2] - detected_workers[j][2]) <= 40 and abs(
                    # detected_workers[i][1] - detected_workers[j][1]) <= 40:
                    duplications.append(detected_workers[j])
                j += 1
            i += 1

        return [x for x in detected_workers if x not in duplications]

    def area_checking(self, refined_workers):
        for worker in range(0, len(refined_workers)):
            height = abs(refined_workers[worker][3] - refined_workers[worker][2])
            width = abs(refined_workers[worker][1] - refined_workers[worker][0])
            center_x = int(width / 2) + refined_workers[worker][0]
            center_y = int(height / 2) + refined_workers[worker][2]
            area = width * height
            refined_workers[worker].append(area)
            refined_workers[worker].append(center_x)
            refined_workers[worker].append(center_y)

        for i in refined_workers:
            for j in range(len(i)):
                if j == 2:
                    if i[j] > 540 and i[-3] < 1200:
                        refined_workers.remove(i)
        return refined_workers

    def worker_tracking(self, tracking_list):
        object_tracker = self.ct.update(tracking_list)
        for (objectID, centeroid) in object_tracker.items():
            text_track = "W {}".format(objectID)
            cv2.putText(self.frame, text_track, (centeroid[0], centeroid[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            cv2.circle(self.frame, (centeroid[0], centeroid[1] + 50), 4, (0, 0, 255), -1)

        for (objectID, centeroid) in object_tracker.items():
            worker_match = []
            worker_min_distance = 50
            for worker_location in tracking_list:
                # distance = math.sqrt(
                #     ((worker_location[-2] - centeroid[0]) ** 2) + ((worker_location[-1] - centeroid[1]) ** 2))
                # if distance < worker_min_distance:
                #     worker_min_distance = distance
                    worker_match = worker_location

            self.util_match_table[objectID] = {"worker_location": worker_match}
        self.util_detection_results[self.frame_index] = self.util_match_table;

    def draw_bounding_box_workers(self, final_refined_workers):

        for i in range(0, len(final_refined_workers)):
            cv2.putText(self.frame, "Person: {}".format(str(len(final_refined_workers))), (10, 20), self.font, 2,
                        (0, 255, 255), 2,
                        cv2.FONT_HERSHEY_SIMPLEX)

            text = "Person: {}%".format(str(final_refined_workers[i][4]))
            (text_width, text_height) = cv2.getTextSize(text, self.font, 1, thickness=1)[0]

            box_coords = (
                (final_refined_workers[i][0] - 10, final_refined_workers[i][2] - 10),
                (
                    final_refined_workers[i][0] + text_width,
                    (final_refined_workers[i][2] - text_height - 10)))

            cv2.rectangle(self.frame, box_coords[0], box_coords[1], self.rectangle_bgr, cv2.FILLED)

            cv2.putText(self.frame, "Person: {}%".format(str(final_refined_workers[i][4])),
                        (final_refined_workers[i][0] - 10, final_refined_workers[i][2] - 10),
                        self.font, 1,
                        (0, 0, 0), cv2.FONT_HERSHEY_PLAIN)

            roi_person = self.frame_rgb[
                         abs(final_refined_workers[i][2] - 10):abs(final_refined_workers[i][3] + 10),
                         abs(final_refined_workers[i][0] - 10):abs(final_refined_workers[i][1]) + 10]

            self.track_list.append([final_refined_workers[i][0], final_refined_workers[i][2],
                                    final_refined_workers[i][1], final_refined_workers[i][3]])

            cv2.rectangle(self.frame,
                          (abs(final_refined_workers[i][0] - 10), abs(final_refined_workers[i][2] - 10)),
                          (final_refined_workers[i][1] + 10, final_refined_workers[i][3] + 10),
                          (0, 255, 0), 1)

    def worker_detection(self, model_name, label_path, video_name, out_put):
        video = cv2.VideoCapture(video_name)
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        out = cv2.VideoWriter(out_put, cv2.VideoWriter_fourcc(*'XVID'),
                              30, (frame_width, frame_height))

        while video.isOpened():
            ret, self.frame = video.read()
            if ret == True:
                self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                if self.frame_index % self.skip_frames == 0:
                    s_regions, s_regions_overlap, m_regions, m_regions_overlap, \
                    l_regions, l_regions_overlap = self.sub_regions(self.frame_rgb, frame_width, frame_height)

                    path_to_check = os.path.join(model_name, 'frozen_inference_graph.pb')
                    categories = label_map_util.convert_label_map_to_categories(
                        label_map_util.load_labelmap(label_path),
                        max_num_classes=self.NUM_WORKER_CLASSES,
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
                            self.detection_results.append(
                                [(xmin + (idx * self.small_width)), (xmax + (idx * self.small_width)), ymin, ymax,
                                 int(acc),
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
                            self.detection_results.append(
                                [(xmin + (idx * self.small_width) + (int(self.small_width / 2))),
                                 (xmax + (idx * self.small_width) + (int(self.small_width / 2))), ymin, ymax, int(acc),
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
                            self.detection_results.append(
                                [(xmin + (idx * self.mid_width)), (xmax + (self.mid_width * idx)),
                                 (ymin + int(frame_height / 4)),
                                 (ymax + int(frame_height / 4)), int(acc),
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
                            self.detection_results.append(
                                [(xmin + (idx * self.mid_width) + (int(self.mid_width / 2))),
                                 (xmax + (idx * self.mid_width) + (int(self.mid_width / 2))),
                                 (ymin + int(frame_height / 4)),
                                 (ymax + int(frame_height / 4)), int(acc),
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
                            self.detection_results.append(
                                [(xmin + (idx * self.large_width)), (xmax + (self.large_width * idx)), (
                                        ymin + int(frame_height / 2)),
                                 (ymax + int(frame_height / 2)), int(acc),
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
                            self.detection_results.append(
                                [(xmin + (idx * self.large_width) + (int(self.large_width / 2))),
                                 (xmax + (idx * self.large_width) + (int(self.large_width / 2))),
                                 (ymin + int(frame_height / 2)),
                                 (ymax + int(frame_height / 2)), int(acc),
                                 classification])

                    if len(self.detection_results) == 0:
                        print("Workers are not detected!")
                    else:
                        refine_workers_detection = self.remove_duplicates(self.detection_results)
                        refined_area_workers = self.area_checking(refine_workers_detection)
                        store_refined_detection = refined_area_workers
                        self.draw_bounding_box_workers(refined_area_workers)
                        self.worker_tracking(self.track_list)
                        out.write(self.frame)
                        del self.track_list[:]
                        self.frame_index += 1
                else:
                    out.write(self.frame)
                    self.frame_index += 1
            else:
                break

        with open("util_detection{}.json".format(video_name), "w") as file:
            j = json.dumps(self.util_detection_results)
            file.write(j)
