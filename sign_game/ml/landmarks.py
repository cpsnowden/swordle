
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pandas as pd
import os
import numpy as np


class Landmarks():
    def __init__(self):
        self.mp_hands = mp.solutions.hands  # hands model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
        self.model = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

    def image_to_landmark(self, frame, draw_landmarks=False):

        # Make detection
        results, frame = self.__mediapipe_detection(frame)
        landmark_object = self.__get_landmark_object(results)
        if draw_landmarks:
            # Draw landmarks using Mediapipe
            frame = self.__draw_landmarks(frame, results)

        return frame, landmark_object

    def __mediapipe_detection(self, image):
        # CV2 gets image as BGR, this converts it to RGB
        rbg_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Locks write on image so that nobody can change the image while we process
        rbg_image.flags.writeable = False
        # This uses mediapipe to detect
        results = self.model.process(rbg_image)
        rbg_image.flags.writeable = True
        # Converts it back to BGR
        bgr_image = cv2.cvtColor(rbg_image, cv2.COLOR_RGB2BGR)
        return results, bgr_image

    def __draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS)

        return image

    def __get_landmark_object(self, results):
        landmark_object = {}
        if results.multi_hand_landmarks:
            detected_hand = results.multi_hand_landmarks[0]
            for handmark in self.mp_hands.HandLandmark:
                landmark = detected_hand.landmark[handmark]
                # ??
                name = str(handmark)[13:]
                landmark_object[name+'_X'] = landmark.x
                landmark_object[name+'_Y'] = landmark.y
                landmark_object[name+'_Z'] = landmark.z

        return landmark_object

    def __get_landmark_np_array(self, results):
        if results.multi_hand_landmarks:
            detected_hand = results.multi_hand_landmarks[0]

            def extract_handmark(handmark):
                landmark = detected_hand.landmark[handmark]
                return (landmark.x, landmark.y, landmark.z)

            return np.array([extract_handmark(handmark) for handmark in self.mp_hands.HandLandmark])

    def image_to_landmark_np(self, frame, draw_landmarks=False):

        # Make detection
        results, frame = self.__mediapipe_detection(frame)

        landmark_object = self.__get_landmark_np_array(results)
        if draw_landmarks:
            # Draw landmarks using Mediapipe
            frame = self.__draw_landmarks(frame, results)

        return frame, landmark_object
