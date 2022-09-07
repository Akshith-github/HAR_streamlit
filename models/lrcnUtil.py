import numpy as np,cv2
from collections import deque

path_to_model = "models\convlstm_model___Date_Time_2022_03_24__09_33_27___Loss_0.8591782450675964___Accuracy_0.6639344096183777.h5"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
class LRCN:
    model = None
    SEQUENCE_LENGTH=20
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

    def model_Loader(self):
        # load lrcn model
        from tensorflow import keras
        self.model = keras.models.load_model(path_to_model)
        return self

    def predict_single_action_on_frames_list(self,frames_list):
        '''
        This function will perform single action recognition prediction on a video using the LRCN model.
        Args:
        video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
        SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
        '''
        # Initialize the VideoCapture object to read from the video file.
        
        LRCN_model = self.model
        
        # Initialize a variable to store the predicted action being performed in the video.
        predicted_class_name = ''


        # Get the number of frames in the video.
        video_frames_count = len(frames_list)
        print(type(video_frames_count),type(self.SEQUENCE_LENGTH))
        # Calculate the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count/self.SEQUENCE_LENGTH),1)

        # Iterating the number of times equal to the fixed length of sequence.
        for frame_counter in range(0,len(frames_list),skip_frames_window):

            # Set the current frame position of the video.

            # Read a frame.
            success, frame = 1,frames_list[frame_counter]

            # Check if frame is not read properly then break the loop.
            if not success:
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255
            
            # Appending the pre-processed frame into the frames list
            frames_list.append(normalized_frame)

        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index.
        predicted_class_name = CLASSES_LIST[predicted_label]
        
        # Display the predicted action along with the prediction confidence.
        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
        return predicted_class_name,{predicted_labels_probabilities[predicted_label]}
    

    def predict_single_action(self,video_file_path):
        '''
        This function will perform single action recognition prediction on a video using the LRCN model.
        Args:
        video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
        SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
        '''
        # Initialize the VideoCapture object to read from the video file.
        video_reader = cv2.VideoCapture(video_file_path)
        LRCN_model = self.model

        # Get the width and height of the video.
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Declare a list to store video frames we will extract.
        frames_list = []
        
        # Initialize a variable to store the predicted action being performed in the video.
        predicted_class_name = ''


        # Get the number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print(type(video_frames_count),type(self.SEQUENCE_LENGTH))
        # Calculate the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count/self.SEQUENCE_LENGTH),1)

        # Iterating the number of times equal to the fixed length of sequence.
        for frame_counter in range(self.SEQUENCE_LENGTH):

            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Read a frame.
            success, frame = video_reader.read() 

            # Check if frame is not read properly then break the loop.
            if not success:
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255
            
            # Appending the pre-processed frame into the frames list
            frames_list.append(normalized_frame)

        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index.
        predicted_class_name = CLASSES_LIST[predicted_label]
        
        # Display the predicted action along with the prediction confidence.
        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
            
        # Release the VideoCapture object. 
        video_reader.release()
    
    def predict_on_video(self,video_file_path, output_file_path, SEQUENCE_LENGTH=20):
        '''
        This function will perform action recognition on a video using the LRCN model.
        Args:
        video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
        output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
        SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
        '''

        # Initialize the VideoCapture object to read from the video file.
        video_reader = cv2.VideoCapture(video_file_path)

        # Get the width and height of the video.
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize the VideoWriter Object to store the output video in the disk.
        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

        # Declare a queue to store video frames.
        frames_queue = deque(maxlen = SEQUENCE_LENGTH)

        # Initialize a variable to store the predicted action being performed in the video.
        predicted_class_name = ''

        # Iterate until the video is accessed successfully.
        while video_reader.isOpened():

            # Read the frame.
            ok, frame = video_reader.read() 
            
            # Check if frame is not read properly then break the loop.
            if not ok:
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255

            # Appending the pre-processed frame into the frames list.
            frames_queue.append(normalized_frame)
            LRCN_model = self.model
            # Check if the number of frames in the queue are equal to the fixed sequence length.
            if len(frames_queue) == SEQUENCE_LENGTH:

                # Pass the normalized frames to the model and get the predicted probabilities.
                predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

                # Get the index of class with highest probability.
                predicted_label = np.argmax(predicted_labels_probabilities)

                # Get the class name using the retrieved index.
                predicted_class_name = CLASSES_LIST[predicted_label]
                yield predicted_class_name,frame
            # Write predicted class name on top of the frame.
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # Write The frame into the disk using the VideoWriter Object.
            video_writer.write(frame)
            
        # Release the VideoCapture and VideoWriter objects.
        video_reader.release()
        video_writer.release()