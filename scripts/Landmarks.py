
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


    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CV2 gets image as BGR, this converts it to RGB
        image.flags.writeable = False # Locks write on image so that nobody can change the image while we process
        results = self.model.process(image) # This uses mediapipe to detect
        image.flags.writeable = True # Unlocks write on image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Converts it back to BGR
        return image, results

    def draw_landmarks(self, image, results):
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS)

        return image

    def get_landmark_object(self, results):
        landmark_object = {}
        if results.multi_hand_landmarks:
            for handmark in self.mp_hands.HandLandmark: 
                landmark = results.multi_hand_landmarks[0].landmark[handmark]
                name = str(handmark)[13:]
                landmark_object[name+'_X'] = landmark.x
                landmark_object[name+'_Y'] = landmark.y
                landmark_object[name+'_Z'] = landmark.z
            
        return(landmark_object)

    def image_to_landmark(self, frame):
   
        # Make detection
        image, results = self.mediapipe_detection(frame)

        if results: # A hand has been found
            # Draw landmarks using Mediapipe
            image = self.draw_landmarks(image, results)
            landmark_object = self.get_landmark_object(results)
            return image, landmark_object

    def get_image_with_landmarks(self, image_path):
    
        image = cv2.imread(image_path)
        image_with_landmarks, landmark_object = self.image_to_landmark(image)
        plt.imshow(image_with_landmarks)
        plt.show()
        print(len(landmark_object.keys()))


    def video_to_landmark(self, video_path):
        array_landmark_objects = []
        cap = cv2.VideoCapture(video_path) # Grab video from file

        while cap.isOpened():
            # Read a feed
            ret, frame = cap.read()
            
            if ret == True:
                _, landmark_object = self.image_to_landmark(frame)

                array_landmark_objects.append(landmark_object)
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()

        print(pd.DataFrame.from_dict(array_landmark_objects).dropna().reset_index(drop=True))

    def create_csv_from_dataset_folder(self):
 
        array_landmark_objects = []
        curr_dir = os.getcwd() 
        img_ds_path = curr_dir + '/asl_dataset'
        dir_folders = os.listdir(img_ds_path)

        for folder_name in dir_folders:
            folder_files = os.listdir(img_ds_path+'/'+folder_name)

            for image_name in folder_files:
                image_path = img_ds_path+'/'+folder_name+'/'+image_name
                image = cv2.imread(image_path)
                image_with_landmarks, landmark_object = self.image_to_landmark(image)

                landmark_object['TARGET'] = folder_name.upper()

                landmark_image_path = curr_dir +'/asl_dataset_landmarks/'+folder_name+'/'+image_name
                cv2.imwrite(landmark_image_path, image_with_landmarks)
                cv2.waitKey(0)
                landmark_object['PATH'] = './asl_dataset_landmarks/'+folder_name+'/'+image_name

                array_landmark_objects.append(landmark_object)

        df = pd.DataFrame.from_dict(array_landmark_objects)

        try:
            df.to_csv('./images_ds.csv')
        except Exception as e:
            print(e)
        else:
            print('CVS file created!')



if __name__ == '__main__':
    image_path = './asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg'
    video_path = './asl_dataset_videos/J/1.avi'
    landmarks = Landmarks()
    # landmarks.get_image_with_landmarks(image_path)
    # landmarks.video_to_landmark(video_path)
    landmarks.create_csv_from_dataset_folder()


