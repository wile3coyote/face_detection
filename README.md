# Facial Recognition using FaceNet

This project was developed to help me better understand the intricacies involved in facial recognition. The project has 4 major stages. First is the extraction of the faces from the image. This done using Multi-task Cascaded Convolutional Networks (MTCNN), which detects individual faces from the image and extracts the facial landmarks(nose, eyes, end points of mouth, and the bounding box). After this, the extracted face images are sent to FaceNet model, which is recreated in Keras. The FaceNet model, then maps the face to a vector, for which the distance directly corresponds to the similarity of the faces. The embeddings are created for all the images in the dataset provided with the labels corresponding to the name of the folder of the dataset. Now an SVM classifier is used to train based on the embeddings. Finally, we use the trained SVM model to predict the identity of the image given as an input. I also implemented a live facial recognition, which using the camera of the computer, to recognise the faces live. This is done using OpenCV, which converts the frames generated by the camera as input to our SVM model to detect and recognise the faces live.

## Setup

The FaceNet model is located in the model folder. For creating the dataset, create a folder called 'data' and categorize all the classes of your dataset in different folders, with each containing the images respectively. The name of the folder is considered the label of the specfic face. After the dataset is created, run load_database.py, which extracts the facial features for all the images available in the datset. This usually takes long, depending on the dataset and is a seperate script. The features are saved as a compressed numpy array, which is used by other scripts. After the embeddings are created, there are two ways of predicting, the first detection.py takes an input image in the root folder, and recognises the faces in the image, the second, video_detection.py uses the camera in your computer and recognises the faces directly.
