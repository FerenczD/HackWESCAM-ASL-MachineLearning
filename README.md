This project is intended to utilize object detection machine learning to interpret american sign language
and display the interpretation in real time on the capture screen.

The official pre-build tensorflow neural network model was used as a basis, and further adapted to work with sign language detection.

Datasets which include hundreds of images of sign language gestures are used from loicmarie's Github library.

Screen captures uses opencv to capture frames, transform them, and pass them to the neural network function.

Edge detection code was not implemented but could be used to potentially increase accuracy.

To use it just execute the main file and your webcam will open, in the screen it will appear a red square, which will be reading the signs. To get the best accuracy is better to have a solid color background and just, display your hand making one of the letter of the American Sign Language alphabet


EXCLUDED FROM UPLOAD:
Dataset files (~75000 files for training)
Tensorboard Logging Data (too large to upload)



Created by: Ferencz Dominguez, Lu Bilyk, Nik Kershaw, Peter Socha
Last Edited: May 6 2018, 10:57AM
