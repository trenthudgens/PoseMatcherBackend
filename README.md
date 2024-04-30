# PoseMatcherBackend
The core functionality and REST API of the PoseMatcher frontend.

# Project Description
PoseMatcher is a human pose comparison tool. Users upload two images of a person to our website, and we detect and compare the human poses to return a similarity score. This tool has potential uses in physical therapy, insurance fraud, exercise science, sports medicine, healthcare, and much more. It is run on a frontend website and a backend server.

## Setup
Setup and Installation Guide
For the easiest user experience follow these instructions:
Open a web browser of your choice
Navigate to the following URL: https://cairob.github.io/PoseMatcherUI/#/home
Backend installation:
Backend is not trivial to setup and run, and requires special hardware and network configuration. However a hosted version of the backend is available at http://unmedicated-person.us/
If you happen to be an insane person and want to run it anyway:
Ensure you have a NVIDIA GPU.
Run the command: git clone https://github.com/trenthudgens/PoseMatcherBackend.git
Extract, and navigate to that directory
Backend was run using the following versions of the core libraries, on a NVIDIA GTX 1070. Use miniconda for dependency management:
Tensorflow, version 2.10
Cudnn version 8.1
See the “backend requirements.txt” document for all other dependencies. You must find the exact versions of Tensorflow and Cudnn for your exact GPU, and specific versions of those dependencies may not work for your GPU, and the model that we used, and may conflict with other required dependencies in the project. Good luck!
server.py provides a provisional function (download_model()) to download the AI model that is used for the project, Metrabs, which is required
Run “flask server.py” in the same directory as all the files in the backend
The backend server is designed to work with Nginx, flask, waitress on top of flask, and a forward port. Modification would be required to be tested locally.

#Frontend User Manual
Open a browser and go to https://cairob.github.io/PoseMatcherUI/#/home
Upload one image to each of the image fields, each image being a photograph of one human person. If some of their body is hidden, that’s OK.
Click submit. After a few seconds, a comparison score and the images will appear.

