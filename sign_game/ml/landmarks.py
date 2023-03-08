
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pandas as pd
import os

class Landmarks():
    def __init__ (self):
        self.mp_hands = mp.solutions.hands # hands model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        self.model = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

    def image_to_landmark(self, frame, draw_landmarks=False):

        # Make detection
        results = self.__mediapipe_detection(frame)
        landmark_object = self.__get_landmark_object(results)
        if draw_landmarks:
            # Draw landmarks using Mediapipe
            frame = self.__draw_landmarks(frame, results)

        return frame, landmark_object

    def __mediapipe_detection(self, image):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CV2 gets image as BGR, this converts it to RGB
        bgr_image.flags.writeable = False # Locks write on image so that nobody can change the image while we process
        results = self.model.process(bgr_image) # This uses mediapipe to detect
        return results

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
                landmark_object[name+'_x'] = landmark.x
                landmark_object[name+'_y'] = landmark.y
                landmark_object[name+'_z'] = landmark.z

        return landmark_object
