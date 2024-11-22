# Image-Colourization-Project
This project demonstrates an image colorization application using Python, OpenCV, and a convolutional neural network (CNN). The project takes grayscale images as input and outputs their colorized versions using a pre-trained model.

Features:
1. Converts grayscale images to colorized versions.
2. Utilizes a pre-trained model for efficient and accurate colorization.
3. Employs OpenCV for image processing and visualization.
   
How It Works
1. Pre-trained Model:
- The project uses a pre-trained model (colorization_release_v2.caffemodel), which can be downloaded https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1.
- This model is based on a deep learning framework and is trained to predict the chrominance (color) channels from a grayscale (luminance) image.

2. Input Processing:
- The input grayscale image is resized and preprocessed to match the input dimensions required by the CNN.
- The luminance channel is extracted and normalized.
  
3. Colorization:
- The pre-trained model predicts the chrominance channels.
- These channels are combined with the original luminance channel to form a colorized image.
  
4. Post-Processing:
- The colorized image is converted back to the BGR color space and scaled to standard image intensity levels.
- The result is displayed using OpenCV.
  
Prerequisites:
1. Python (>=3.6)
2. OpenCV (>=4.0)
3. Numpy

Usage:
- Run the script to colorize an image:
python colorize.py --image <path-to-grayscale-image>

Acknowledgments:
1. The pre-trained model used in this project is developed by Richard Zhang et al.
2. Model source: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1.
3. Special thanks to OpenCV for simplifying image processing tasks.
