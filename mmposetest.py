from mmpose.apis import MMPoseInferencer
import numpy as np
import matplotlib
matplotlib.use('Agg') #Agg backend for server applications
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def getKeypoints(inferencer, image_path):
    """
    Uses MMPose's AI to generate a 2D pose from the image.
    If multiple people are in the images, multiple poses will be created
    
    """
    result_generator = inferencer(image_path, out_dir='output')
    result = next(result_generator)
    keypoints = result['predictions'][0][0]['keypoints']
    keypointscores = result['predictions'][0][0]['keypoint_scores'] #the score tells us level of certainty for each point
    """
    If we want, we can extract multiple poses.
    If we want, we can extract the scores of each keypoint if the 2D model was selected.
    for instance in result['predictions'][0][0]['keypoints']: #grabs all people in the image
        keypoints.append(instance['keypoints'])
        #)
    """
    keypoints = np.array(keypoints)
    return keypoints,keypointscores


def flip(keypoints):
    """
    reverses the image and swaps left and right points.
    """
    #labels = ['nose', 'eye', 'eye', 'ear', 'ear', 'shoulder', 'shoulder', 'hand',
              #'hand', 'elbow', 'elbow', 'hip', 'hip', 'knee', 'knee', 'foot', 'foot']
    #flip nodes over x axis
    center = np.mean(keypoints, axis=0)
    flipx = np.copy(keypoints)
    flipx[:, 0] = center[0] - (flipx[:, 0] - center[0])
    #swap left and right keypoints
    for i in range(1, len(flipx) - 1, 2):
        temp1 = flipx[i][0]
        temp2 = flipx[i][1]
        flipx[i][0] = flipx[i+1][0]
        flipx[i][1] = flipx[i+1][1]
        flipx[i+1][0] = temp1
        flipx[i+1][1] = temp2    
    return flipx


def scale(X,Y):
    """
    #scale and center Y to X's size and position
    """
    centerX = np.mean(X, axis=0)
    centerY = np.mean(Y, axis=0)
    
    scale_x = (max(X[:, 0]) - min(X[:, 0])) / (max(Y[:, 0]) - min(Y[:, 0]))
    scale_y = (max(X[:, 1]) - min(X[:, 1])) / (max(Y[:, 1]) - min(Y[:, 1]))
    scaled_Y = (Y - centerY) * np.array([scale_x, scale_y]) + centerX
    return scaled_Y

def OP(X, Y):
    Y = scale(X,Y)
    #Procrustes Y
    centerX = np.mean(X, axis=0)
    centerY = np.mean(Y, axis=0)
    centered_X = X - centerX
    centered_Y = Y - centerY
    cov_matrix = np.dot(centered_Y.T, centered_X)
    U, S, Vt = np.linalg.svd(cov_matrix)
    rotation_matrix = np.dot(U, Vt)
    """
    #make sure procrustes does not flip points
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] *= -1
    """
    pro_Y = np.dot(centered_Y, rotation_matrix) + centerX
    return pro_Y


def RMSE(source_pred, test_pred):
    """
    Root Mean Squared Error
    Calculates the total error between the poses and returns an accuracy score.
    """
    #scale and center test_pred to source_pred's size and position
    centroidX = np.mean(source_pred, axis=0)
    centroidY = np.mean(test_pred, axis=0)
    scale_x = (max(source_pred[:, 0]) - min(source_pred[:, 0])) / (max(test_pred[:, 0]) - min(test_pred[:, 0]))
    scale_y = (max(source_pred[:, 1]) - min(source_pred[:, 1])) / (max(test_pred[:, 1]) - min(test_pred[:, 1]))
    scaled_test = (test_pred - centroidY) * np.array([scale_x, scale_y]) + centroidX
    
    #find max_RMSE
    max_x = max(np.max(source_pred[:, 0]), np.max(test_pred[:, 0]))
    max_y = max(np.max(source_pred[:, 1]), np.max(test_pred[:, 1]))
    min_x = min(np.min(source_pred[:, 0]), np.min(test_pred[:, 0]))
    min_y = min(np.min(source_pred[:, 1]), np.min(test_pred[:, 1]))
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_RMSE = np.sqrt(range_x**2 + range_y**2)
    
    #calculate accuracy
    squared_diff = np.square(source_pred - scaled_test)
    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)
    accuracy = (1 - rmse/max_RMSE) * 100
    return accuracy


def SAE(source_pred, test_pred):
    
    """
    Square Average Error
    Creates a square containing all the datapoints.
    Sums the error of the difference between source_pred and test_pred
    against the maximum error within the square.
    Input: two scaled, centered sets of 2D keypoints.
    Output: accuracy score 0-100.
    """
    #find the distances of each point
    test = np.copy(test_pred)
    distances = np.sqrt(np.sum((source_pred - test)**2,axis=1))
    
    #find the max distance from test_pred to the corners
    #of the square containing all points
    max_dist = []
    max_x = max(np.max(source_pred[:, 0]), np.max(test[:, 0]))
    max_y = max(np.max(source_pred[:, 1]), np.max(test[:, 1]))
    min_x = min(np.min(source_pred[:, 0]), np.min(test[:, 0]))
    min_y = min(np.min(source_pred[:, 1]), np.min(test[:, 1]))
    for keypoint in test:
       dist1 = np.sqrt(np.sum((keypoint - np.array([min_x, min_y]))**2))
       dist2 = np.sqrt(np.sum((keypoint - np.array([min_x, max_y]))**2))
       dist3 = np.sqrt(np.sum((keypoint - np.array([max_x, min_y]))**2))
       dist4 = np.sqrt(np.sum((keypoint - np.array([max_x, max_y]))**2))
       max_distance = max(dist1, dist2, dist3, dist4)
       max_dist.append(max_distance)
    
    #find the average error
    avg_err = np.mean(distances/max_dist)
    accuracy = (1 - avg_err)*100
    return accuracy


def plot_two_keypoints(source_keypoints, test_keypoints, image_path):
    #Plots both sets of keypoints on the image
    if source_keypoints.shape[1] == 2:
        """2D keypoints"""
        fig,ax = plt.subplots()
        image = plt.imread(image_path)
        ax.imshow(image)
        
        #labels = ['nose', 'eye', 'eye', 'ear', 'ear','shoulder','shoulder','hand',
                   #'hand','elbow','elbow','hip','hip','knee','knee','foot','foot']
        # Scatter plot the data points
    
        for i, (x, y) in enumerate(source_keypoints):
            ax.scatter(x, y, color='blue')  # Label points with numbers
        for i, (x, y) in enumerate(test_keypoints):
            ax.scatter(x, y, color='green')
        
        # Label the axes
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        
        # Add a legend
        legend_labels = ["Source Image", "Test Image"]
        legend_colors = ['blue', 'green']
        legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
        plt.legend(handles=legend_patches, loc='upper left')  # Adjust the 'loc' parameter as needed
        
        #disable grid
        plt.axis('off')
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        # Show the 2D plot
        #plt.show()
        
        #set image name
        file_name = os.path.basename(image_path)
        base_name, file_extension = os.path.splitext(file_name)
        new_file_name = f"{base_name}_score{file_extension}"
        #save to the visualizations directory
        subdirectory = "output/visualizations"
        output_path = os.path.join(os.getcwd(), subdirectory)
        os.makedirs(output_path, exist_ok=True)
        #make sure images with the same name are not overwritten
        if os.path.exists(os.path.join(output_path, new_file_name)):
            new_file_name = f"{base_name}_score2{file_extension}"
        plt.savefig(os.path.join(output_path, new_file_name))
        return new_file_name


def plot_scores(keypoints, keypointscores, image_path):
    #Plots the keypoints on the image with score colors
    if keypoints.shape[1] == 2:
        """2D keypoints"""
        fig,ax = plt.subplots()
        image = plt.imread(image_path)
        #ax = fig.add_subplot(111)
        ax.imshow(image)
        
        colors = []
        for score in keypointscores:
            if score < 0.25:
                colors.append('red')
            elif score < 0.5:
                colors.append('orange')
            elif score < 0.75:
                colors.append('gold')
            elif score < 1.0:
                colors.append('palegreen')
            else:
                colors.append('green')
        labels = ['nose', 'eye', 'eye', 'ear', 'ear','shoulder','shoulder','hand',
                  'hand','elbow','elbow','hip','hip','knee','knee','foot','foot']
                
        # Scatter plot the data points
    
        for i, (x, y) in enumerate(keypoints):
            ax.scatter(x, y, color=colors[i], label=labels[i])  # Label points with numbers

        # Label the axes
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        
        # Add a legend
        legend_labels = ["1.0", "0.99 - 0.75", "0.74 - 0.5", "0.49 - 0.25", "< 0.25"]
        legend_colors = ['green', 'palegreen', 'gold', 'orange', 'red']
        legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
        plt.legend(handles=legend_patches, loc='upper left')  # Adjust the 'loc' parameter as needed
        #invert image
        #plt.gca().invert_yaxis()

        # Show the 2D plot
        #plt.show()
        #remove grid
        plt.axis('off')
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        #set image name
        file_name = os.path.basename(image_path)
        base_name, file_extension = os.path.splitext(file_name)
        new_file_name = f"{base_name}_overlay{file_extension}"
        #save to the visualizations directory
        subdirectory = "output/visualizations"
        output_path = os.path.join(os.getcwd(), subdirectory)
        os.makedirs(output_path, exist_ok=True)
        #make sure images with the same name are not overwritten
        if os.path.exists(os.path.join(output_path, new_file_name)):
            new_file_name = f"{base_name}_overlay2{file_extension}"
        plt.savefig(os.path.join(output_path, new_file_name))
        return new_file_name


def analyze(img_path, img_path2):
    """
    Uses MMPose to detect human poses and produce 2D keypoints.
    Analyzes keypoint scores.
    Compares the keypoints and calculates accuracy scores.
    Input: two image files
    Output: 2 keypoint-overlaid images, 2 keypoint-scored images,
    1 image with both sets of keypoints overlaid, and an accuracy score.
    """
    image_names = []
    inferencer = MMPoseInferencer('human')
    #save keypoint-overlaid images and get keypoints
    source_pred, source_scores = getKeypoints(inferencer, img_path)
    test_pred, test_scores = getKeypoints(inferencer, img_path2)
    image_names.append(os.path.basename(img_path))
    image_names.append(os.path.basename(img_path2))
    #analyze score quality
    if np.mean(source_scores) < 0.5:
        print("Select a better source image.")
    if np.mean(test_scores) < 0.5:
        print("Select a better test image.")
    #save visualized score quality
    p1 = plot_scores(source_pred, source_scores, img_path)
    p2 = plot_scores(test_pred, test_scores, img_path2)
    image_names.append(p1)
    image_names.append(p2)
    #test flipped image
    flip_pred = flip(test_pred)
    #calculate OP
    aligned_test = OP(source_pred, test_pred)
    aligned_test_flip = OP(source_pred, flip_pred)
    #compare images with SAE and RMSE
    scores = []
    A1 = SAE(source_pred, aligned_test)
    A2 = RMSE(source_pred, aligned_test)
    A1flip = SAE(source_pred, aligned_test_flip)
    A2flip = RMSE(source_pred, aligned_test_flip)
    if np.mean([A1, A2]) > np.mean([A1flip, A2flip]):
        p3 = plot_two_keypoints(source_pred, aligned_test, img_path)
        image_names.append(p3)
        scores.append(A1)
        scores.append(A2)
        scores = np.round(scores, 2).tolist()
        return scores, image_names
    else:
        p3 = plot_two_keypoints(source_pred, aligned_test_flip, img_path)
        image_names.append(p3)
        scores.append(A1flip)
        scores.append(A2flip)
        scores = np.round(scores, 2).tolist()
        return scores, image_names
    """
if __name__ == "__main__":
    analyze("test images/salute.jpg", "test images/selfie.jpg")
    """