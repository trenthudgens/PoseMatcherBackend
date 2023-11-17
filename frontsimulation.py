import base64
import sys

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        encoded_image_str = encoded_image.decode("utf-8")
    return encoded_image_str

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python frontsimulation.py <source_image_path> <test_image_path>")
        sys.exit(1)

    source_image_path = sys.argv[1]
    test_image_path = sys.argv[2]

    source_base64 = image_to_base64(source_image_path)
    test_base64 = image_to_base64(test_image_path)

    with open("encoded_images.txt", "w") as output_file:
        output_file.write(f"{{\"source\": \"{source_base64}\", \"test\": \"{test_base64}\"}}")
        
"""
Start up backend:
1. open miniconda3's powershell prompt
2. conda activate openmmlab
3. navigate to PoseMatcherBackend's file
4. python PoseMatcherBackendDemo.py
A server address should display (for me it is 127.0.0.1:5000)

How to run this file:
1. open command prompt
2. navigate to PoseMatcherBackend's file
3. python frontsimulation.py testimages/salute.jpg testimages/selfie.jpg
4. curl -X POST -H "Content-Type: application/json" -d @encoded_images.txt http://127.0.0.1:5000/upload

This sends a Json file with the two images to the backend server.
The backend processes the values and returns it to the command prompt.
Copy the resulting image data (big string of letters) into the HTML file image_verification in the templates folder.
Note that the images will be renamed source.jpg and test.jpg. This is important for mmposetest.py
"""