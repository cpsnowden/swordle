
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pandas as df

class Landmarks():
    def __init__ (self):
        self.mp_hands = mp.solutions.hands # hands model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        self.model = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)


    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CV2 gets image as BGR, this converts it to RGB
        image.flags.writeable = False # Locks write on image so that nobody can change the image while we process
        results = self.model.process(image) # This uses mediapipe to detect
        image.flags.writeable = True # Unlocks write on image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Converts it back to BGR
        return image, results

    def draw_landmarks(self, image, results):
        
        # # image_with_landmarks = self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        # image_with_landmarks = self.mp_drawing.draw_landmarks(image,results.multi_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        # return image_with_landmarks
        if results.multi_hand_landmarks:
            # for handmark in self.mp_hands.HandLandmark: 
            #     print(results.multi_hand_landmarks[0].landmark[handmark])
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS)
        # print(len(self.mp_hands.HandLandmark))
        return image

    def get_landmark_object(self, results):
        landmark_object = {}
        if results.multi_hand_landmarks:
            for handmark in self.mp_hands.HandLandmark: 
                landmark = results.multi_hand_landmarks[0].landmark[handmark]
                name = str(handmark)[13:]
                landmark_object[name+'_x'] = landmark.x
                landmark_object[name+'_y'] = landmark.y
                landmark_object[name+'_z'] = landmark.z
            
        return(landmark_object)

    def image_to_landmark(self, frame):
   
        # Make detection
        image, results = self.mediapipe_detection(frame)
        # plt.imshow(image)
        # plt.show()
        # print(results.multi_hand_landmarks)

        # Draw landmarks using Mediapipe
        image = self.draw_landmarks(image, results)
        landmark_object = self.get_landmark_object(results)

        return image, landmark_object
    
    def get_image_with_landmarks(self, image):
    
        # image, _ = 
        image_with_landmarks, landmark_object = self.image_to_landmark(image)
        plt.imshow(image_with_landmarks)
        plt.show()
        print(len(landmark_object.keys()))


# def video_to_landmark(video_path):
#     cap = cv2.VideoCapture(0) # Grab video device 0, which should be the webcam

#     # Define midiapipe hands model
#     with mp_hands.hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#         while cap.isOpened():
#             # Read a feed
#             ret, frame = cap.read()
            
#             # Make detection
#             image, results = mediapipe_detection(frame, hands)
#             # print(results)

#             # Draw landmarks using Mediapipe
#             draw_landmarks(image, results)

#             # Show to screen
#             cv2.imshow('Read Feed', image)
            
#             # Break gracefully
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#         cap.release()
#         cv2.destroyAllWindows()



if __name__ == '__main__':
    image = cv2.imread('./asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg')
    landmarks = Landmarks()
    landmarks.get_image_with_landmarks(image)

