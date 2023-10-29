#Pose Matcher Backend Demo

#all HTML files must be placed in the templates folder
#all uploaded images are stored in the uploads folder

from flask import Flask, render_template, request, redirect, url_for
import os
import json
import base64
import mmposetest

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VISUALIZATIONS_FOLDER'] = 'output/visualizations'

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
            scores, image_names = mmposetest.analyze(sourceImageName, testImageName)

            # Redirect to the 'result' page with the image names and similarity score as URL parameters
            return redirect(url_for('result', score=json.dumps(scores), image=json.dumps(image_names)))

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

if __name__ == '__main__':
    app.run(debug=True)
