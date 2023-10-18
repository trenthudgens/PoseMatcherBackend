#Pose Matcher Backend Demo

#all HTML files must be placed in the templates folder
#all uploaded images are stored in the uploads folder

from flask import Flask, render_template, request, redirect, url_for
from processor import Processor
import os
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Default page with a link to the upload page
@app.route('/')
def index():
    return render_template('home.html')

# This route allows users to upload an image
@app.route('/upload', methods=['POST', 'GET'])
def upload_images():
    #handles data submission
    if request.method == 'POST':
        if 'source' in request.files and 'test' in request.files:
            sourceImage = request.files['source']
            testImage = request.files['test']
            
        if sourceImage and testImage:
            # Save the uploaded images
            sourceImageName, testImageName = save_uploaded_images(sourceImage, testImage)
            # Process images and calculate the similarity score
            p = Processor()
            similarity_score = round(p.mse(sourceImageName, testImageName), 2)
            # Redirect to the 'result' page with the image names and similarity score as URL parameters
            return redirect(url_for('result', score=similarity_score, sourceImageName=sourceImageName, testImageName=testImageName))

    return render_template('upload.html')

def save_uploaded_images(SourceImage, TestImage):
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    #we can save the images as a PNG or a JPEG
    SourceImageName = os.path.join(app.config['UPLOAD_FOLDER'], 'source.jpg')
    TestImageName = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')

    SourceImage.save(SourceImageName)
    TestImage.save(TestImageName)

    return SourceImageName, TestImageName

@app.route('/result')
def result():
    #displays the similarity score and the images
    similarity_score = request.args.get('score')
    sourceImageName = request.args.get('sourceImageName')
    testImageName = request.args.get('testImageName')
    #open images from path (Pillow might be a better way)
    with open(sourceImageName, 'rb') as sourceImage:
        sourceData = base64.b64encode(sourceImage.read()).decode('utf-8')
    with open(testImageName, 'rb') as testImage:
        testData = base64.b64encode(testImage.read()).decode('utf-8')
    return render_template('result.html', score=similarity_score, source=sourceData, test=testData)

if __name__ == '__main__':
    app.run(debug=True)
