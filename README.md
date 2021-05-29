# Face-Mask Detector

Real time face-mask detection using Deep Learning and OpenCV

# About Project

This project uses Deep Neural Network, more specifically a Convolution Neural Network to differentiate between images of people with and without masks. The CNN manages to get a training accuracy of **99.93%**.Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task.

**This application is a kind of alert system which detects the facemask in camera and if some person doesn’t wear a mask, then the authorities are alerted via email regarding that.**
