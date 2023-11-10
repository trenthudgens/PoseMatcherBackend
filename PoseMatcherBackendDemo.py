"""
Pose Matcher Backend Demo
Uploaded images are stored in the uploads folder and removed at the end.
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import mmposetest
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app) #Allows HTTP requests from anywhere.
#CORS(app, resources={r"/*": {"origins": "https://cairob.github.io"}})
app.config['UPLOAD_FOLDER'] = 'uploads'


"""Default page with a link to the frontend."""
@app.route('/')
def index():
    return render_template('home.html')

"""Handles image submission."""
@app.route('/upload', methods=['POST', 'GET'])
def upload_images():
    if request.method == 'POST':
        try:
            # Extract images
            data = request.get_json()
            sourceImage = data.get('source', None)
            testImage = data.get('test', None)
            if sourceImage and testImage:
                # Save images
                sourceImageName, testImageName = save_uploaded_images(sourceImage, testImage)
                # Analyze images and calculate similarity score
                score, imageData = mmposetest.analyze()
                clear_upload_folder()
                return jsonify({'score': score, 'image': imageData})
        except Exception as e:
            error_message = f"Error processing images: {str(e)}"
            print(error_message)
            return jsonify({'error': error_message}), 500
            
"""Decodes base64 images and saves them."""
def save_uploaded_images(sourceBase64, testBase64):
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        # Decode base64 strings
        sourceImageData = base64.b64decode(sourceBase64)
        testImageData = base64.b64decode(testBase64)
        # Save images
        with Image.open(BytesIO(sourceImageData)) as sourceImage:
            sourceImageName = os.path.join(app.config['UPLOAD_FOLDER'], 'source.jpg')
            sourceImage.save(sourceImageName)
        with Image.open(BytesIO(testImageData)) as testImage:
            testImageName = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
            testImage.save(testImageName)
        return sourceImageName, testImageName
    except Exception as e:
        print(f"Error saving images: {str(e)}")
        return None, None


def clear_upload_folder():
    """Deletes contents of the upload folder."""
    """https://www.askpython.com/python/examples/delete-contents-of-folders"""
    extension = ".jpg"  # set the file extension to delete
    for filename in os.listdir(app.config['UPLOAD_FOLDER']): 
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) and filename.endswith(extension):  
                os.remove(file_path)
        except Exception as e: 
            print(f"Error deleting {file_path}: {e}")

if __name__ == '__main__':
    app.run(debug=True)

"""
app.config['VISUALIZATIONS_FOLDER'] = 'output/visualizations'
def save_uploaded_images(SourceImage, TestImage):
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    SourceImageName = os.path.join(app.config['UPLOAD_FOLDER'], 'source.jpg')
    TestImageName = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
    SourceImage.save(SourceImageName)
    TestImage.save(TestImageName)
    return SourceImageName, TestImageName

@app.route('/result')
def result():
    #displays the similarity score and the images
    similarity_scores = json.loads(request.args.get('score'))
    image_names = json.loads(request.args.get('image'))
    if not os.path.exists(app.config['VISUALIZATIONS_FOLDER']):
        os.makedirs(app.config['VISUALIZATIONS_FOLDER'])
    #open images from path (Pillow might be a better way)
    images = []
    for i in range(len(image_names)):
        file_name = os.path.join(app.config['VISUALIZATIONS_FOLDER'], image_names[i])
        with open(file_name, 'rb') as image:
            data = base64.b64encode(image.read()).decode('utf-8')
            images.append(data)
    
    return render_template('result.html',
    SAEscore=similarity_scores[0] if similarity_scores and len(similarity_scores) > 0 else None,
    RMSEscore=similarity_scores[1] if similarity_scores and len(similarity_scores) > 1 else None,
    MMPose1=images[0] if images and len(images) > 0 else None,
    MMPose2=images[1] if images and len(images) > 1 else None,
    score_quality1=images[2] if images and len(images) > 2 else None,
    score_quality2=images[3] if images and len(images) > 3 else None,
    double_plot=images[4] if images and len(images) > 4 else None)
"""

    
"""
Optimizations:
    handle operations asynchronously
    use logging with error messages
"""
