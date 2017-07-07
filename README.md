# Skateboard_Trick_Identifier

# Business Understanding
A skateboarder can identify which trick someone is doing in a photograph. I can see this potentially moving in towards identifying video. Nike SB has a app that is strictly for skateboarding, future iterations of this idea could become a selling point to Nike, Adidas, New Balance, or any other traditional skateboarding company.

# Data Understanding
The data will be gathered by by taking photos of individuals doing 3 different tricks(ollie, kickflip, pop shuv-it ). A group of 4 individuals, including myself, will be performing each of these tricks 100 times. Two cameras will be set up at different angles to be able to capture double the amount of pictures as well as get photographs at varying angles and capture to prevent overfitting.
Another option is to capture video and pull still frames from those videos as the training data set for the tricks. I may be able to decrease the amount of tricks being done because I’d be able to capture more frames (5 frames at once vs. 1 frame). I would be using 2 different cameras to capture the tricks. Something to note is the resolution of the video, HD.
Photos will be processed to have lower resolution to speed up the training time and to fit within the parameters of what Inception-V3 can handle.

# Data Preparation
The pre-trained CNN that I will be using will be from Google, Inception-V3. Inception-V3 can deal with images that are 299x299, but I will be processing the photos to fit this parameter.

# Modeling
Use a convolutional neural network to train the machine to recognize what tricks are being done in the picture. I will be using a pre-trained CNN from google. An appropriate pre-trained CNN would be inception-V3 from Google. This CNN was trained on the data that was available for the ImageNet Competition from 2012. I am choosing to use this because it is easily accessible from TensorFlow and it has been trained on a deep pool of images, first pool filters and second pool filters seem to be well trained. I will be using transfer learning to train the CNN on my photos that I had taken.
The classification will be done training a SVM(Support Vector Machine). From the features that were produced from the CNN, I will push those features to train on a SVM for classification purposes.

# Evaluation
The model will show the percentage of confidence that the machine will be able to correctly classify the trick.
