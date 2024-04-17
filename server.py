from flask import Flask, request, jsonify
from flask_cors import CORS
from pprint import pprint
from waitress import serve

app = Flask(__name__)
CORS(app)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

import logging

logger = logging.getLogger('waitress')
logger.setLevel(logging.DEBUG)

# Enable console logging
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

import tensorflow as tf
import numpy as np
import os
import base64

from matplotlib_visualizer import visualize
from MJPE import *

print(tf.__version__)
print(tf.config.list_physical_devices())


def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}_20211019.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    print(model_path)
    return model_path
# download_model("metrabs_eff2l_y4")


def decode(request, img_name):
    image_b64 = request.json[img_name]
    image_bits = base64.b64decode(image_b64)
    image = tf.image.decode_image(image_bits)[:, :, :3]

    return image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


joint_names =  ['pelv', 'lhip', 'rhip', 'spi1', 'lkne', 'rkne', 'spi2', 'lank', 
                'rank', 'spi3', 'ltoe', 'rtoe', 'neck', 'lcla', 'rcla', 'head', 
                'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri', 'lhan', 'rhan'] # for smpl_24
joint_edges = [
    [1, 4], [1, 0], [2, 5], [2, 0], [3, 6], [3, 0], [4, 7], [5, 8],
    [6, 9], [7, 10], [8, 11], [9, 12], [12, 13], [12, 14], [12, 15],
    [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]
]

# define skeleton for inference and print the joint names
skeleton = 'smpl_24'

tf.device('/GPU:0')
model = tf.saved_model.load("C:/Users/Trent Hudgens/.keras/models/metrabs_eff2l_y4") # metrabs_eff2l_y4


@app.route('/infer', methods=['OPTIONS', 'POST'])
def handle_infer():
    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    elif request.method == 'POST':
        print(request.headers)
        image1 = decode(request, "image1")
        image2 = decode(request, "image2")
        pred1 = model.detect_poses(image1, skeleton=skeleton)
        pred2 = model.detect_poses(image2, skeleton=skeleton)


        poses3d1, poses3d2 = np.array(pred1['poses3d'][0], np.float32), np.array(pred2['poses3d'][0], np.float32)

        # resp_dict = {"boxes":pred['boxes'].numpy().tolist(),
        #              "poses2d":pred['poses2d'].numpy().tolist(),
        #              "poses3d":pred['poses3d'].numpy().tolist()}

        visualize(
            image1.numpy(), 
            pred1['boxes'].numpy(),
            pred1['poses3d'].numpy(),
            pred1['poses2d'].numpy(),
            model.per_skeleton_joint_edges['smpl_24'].numpy(),
            "pose1.png")

        visualize(
            image2.numpy(), 
            pred2['boxes'].numpy(),
            pred2['poses3d'].numpy(),
            pred2['poses2d'].numpy(),
            model.per_skeleton_joint_edges['smpl_24'].numpy(),
            "pose2.png")

        return jsonify({"MJPE": float(normalize_align_and_mjpe(poses3d1, poses3d2)), 
                        "viz1": encode_image("pose1.png"),
                        "viz2": encode_image("pose2.png")})

        return jsonify({"MJPE": 100.0})
        # return preda

        # viz = poseviz.PoseViz(joint_names, joint_edges)

        # # Update the visualization
        # viz.update(
        #     frame=image,
        #     # frame=np.zeros([128, 128, 3], np.uint8),
        #     boxes=np.array(boxes, np.float32),
        #     poses=poses3d,
        #     camera=cameralib.Camera.from_fov(55, image.shape[:2]))
    
@app.route('/test', methods=['GET'])
def handle_options():
    return "HELLO!!!"
if __name__ == '__main__':

    print("test")
    serve(app, host='0.0.0.0', port=25565, threads=6)