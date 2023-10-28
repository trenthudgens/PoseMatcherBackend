from mmpose.apis import MMPoseInferencer
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes, norm

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
    centerX = np.mean(X, axis=0)
    centerY = np.mean(Y, axis=0)
    #scale and center Y to X's size and position
    scale_x = (max(X[:, 0]) - min(X[:, 0])) / (max(Y[:, 0]) - min(Y[:, 0]))
    scale_y = (max(X[:, 1]) - min(X[:, 1])) / (max(Y[:, 1]) - min(Y[:, 1]))
    scaled_Y = (Y - centerY) * np.array([scale_x, scale_y]) + centerX
    return scaled_Y


def OP(X, Y):
    #Procrustes Y (rotates Y to X. May flip data)
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
    

def OP2(source_pred, test_pred):
    """
    Orthogonal Procrustes
    centers, scales, flips, rotates, then alignes test pred with source pred.
    using code from
    https://www.programcreek.com/python/?code=Relph1119%2FGraphicDesignPattern
    ByPython%2FGraphicDesignPatternByPython-master%2Fvenv%2FLib%2Fsite-package
    s%2Fscipy%2Flinalg%2Ftests%2Ftest_procrustes.py
    """
    def _centered(A):
        mu = A.mean(axis=0)
        return A - mu, mu
    T, T_mu = _centered(test_pred) #center on the origin
    S, S_mu = _centered(source_pred) #center on the origin
    R,s = orthogonal_procrustes(T, S) #finds optimal rotation and reflection matrix
    scale = s/np.square(norm(T)) #scales T to S's size
    op_test = scale * np.dot(T,R) + S_mu #applies OP to T and translates to source_pred's position
    return op_test


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
    #mean = np.mean(distances)
    """
    
    #weigh outliers
    outliers = distances > 2 * mean
    if any(outliers):
        print("outliers present")
        center = np.mean(source_pred)
        for i in np.where(outliers)[0]:
            # Calculate the vector from source_pred[i] to the center
            vector = center - source_pred[i]
            # Normalize the vector
            normalized_vector = vector / np.linalg.norm(vector)
            # Update the outlier to be mean distance away from source_pred[i] on the line to the center
            test[i] = source_pred[i] + mean * normalized_vector
        #replace outlier with average distance
        distances[outliers] = mean
        mean = np.mean(distances) #update mean
        print("updated mean:",mean)
    """
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add a legend
        #ax.legend()
    
        # Show the 2D plot
        plt.show()


def plot_keypoints(keypoints, keypointscores, image_path):
    #Plots the keypoints on the image with score colors
    """3D keypoints"""
    """
    if keypoints.shape[1] == 3:
        fig,ax = plt.subplots()
        image = plt.imread(image_path)
        ax.imshow(image)
        
        #no score given for keypoints
        labels = ['center', 'hip', 'knee', 'foot', 'hip','knee','foot','stomach','chest',
                  'head','head','shoulder','elbow','hand','shoulder','elbow','hand']
    
        # Scatter plot the data points X and Y
        for i, (x, y, z) in enumerate(keypoints):
            #ax.scatter(x, y, z, color='red', label=labels[i])
            if i == 0: #center
                ax.scatter(x, y, color='black', label=labels[i])
            if 1 <= i <= 3: #left leg
                ax.scatter(x, y, color='blue', label=labels[i])
            if 4 <= i <= 6: #right leg
                ax.scatter(x, y, color='purple', label=labels[i])
            if 7 <= i <= 8: #spine
                ax.scatter(x, y, color='green', label=labels[i])
            if 9 <= i <= 10: #head
                ax.scatter(x, y, color='black', label=labels[i])
            if 11 <= i <= 13: #right arm
                ax.scatter(x, y, color='orange', label=labels[i])
            if 14 <= i <= 16: #left arm
                ax.scatter(x, y, color='red', label=labels[i])
                
        # Label the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
    
        # Add a legend
        ax.legend()
    
        #ax.view_init(elev=30, azim=70)  # Set elevation and azimuth angles
    
    
        # Show the 3D plot
        plt.show()
        """
    
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add a legend
        ax.legend()
        #invert image
        #plt.gca().invert_yaxis()
    
        # Show the 2D plot
        plt.show()
        
        #save the image with keypoints overlaid
        #plt.savefig('output_image.jpg')

def main():
    img_path = "C:/Users/Eric Ratzleff/PoseMatcherBackend/test images/salute.jpg"
    img_path2 = "C:/Users/Eric Ratzleff/PoseMatcherBackend/test images/spinal.jpg"
    inferencer = MMPoseInferencer('human') #2D keypoints
    #inferencer = MMPoseInferencer(pose3d="human3d") #3D keypoints
    source_pred, source_scores = getKeypoints(inferencer, img_path)
    test_pred, test_scores = getKeypoints(inferencer, img_path2)
    if np.mean(source_scores) < 0.5:
        print("Select a better source image.")
    if np.mean(test_scores) < 0.5:
        print("Select a better test image.")
    plot_keypoints(source_pred, source_scores, img_path)
    plot_keypoints(test_pred, test_scores, img_path2)
    
    flipx = flip(test_pred)
    
    scaledx = scale(source_pred, test_pred) #best for same
    scaledxflip = scale(source_pred, flipx) #best for flipped
    
    alignedx = OP(source_pred, scaledx) #best for rotate
    alignedxflip = OP(source_pred, scaledxflip) #best for flipped on rotate
    
    alignedx2 = OP2(source_pred, scaledx)
    alignedxflip2 = OP2(source_pred, scaledxflip)
    
    plot_two_keypoints(source_pred, alignedx, img_path)
    plot_two_keypoints(source_pred, alignedxflip, img_path)
    plot_two_keypoints(source_pred, alignedx2, img_path)
    plot_two_keypoints(source_pred, alignedxflip2, img_path)
    
    print("OP1:")
    print("Accuracy using SAE:", SAE(source_pred, alignedx))
    print("Accuracy using SAE:", SAE(source_pred, alignedxflip))
    print("Accuracy using RMSE:", RMSE(source_pred, alignedx))
    print("Accuracy using RMSE:", RMSE(source_pred, alignedxflip))
    print("OP2:")
    print("Accuracy using SAE:", SAE(source_pred, alignedx2))
    print("Accuracy using SAE:", SAE(source_pred, alignedxflip2))
    print("Accuracy using RMSE:", RMSE(source_pred, alignedx2))
    print("Accuracy using RMSE:", RMSE(source_pred, alignedxflip2))

    

if __name__ == "__main__":
    main()
    
    """
    Results:
    black and white
    SAE: 62
    RMSE: 78
    warrior 4 vs warrior 6
    SAE: 95
    RMSE:96
    warrior 4 rotate vs warrior 4 flip
    SAE: 99
    RMSE: 99
    warrior 9 vs warrior 2
    SAE: 96
    RMSE: 97
    salute vs selfie
    SAE: 95
    RMSE: 96
    frontflip vs extended
    SAE: 66
    RMSE: 83
    trent2 vs trent3
    SAE: 87
    RMSE: 92
    trent3 vs trent2
    SAE: 83
    RMSE:84
    
    Bad poses are below 80
    Average poses get 85-90
    Great poses get around 95
    Identical poses get 99
    
    We can reject images if the test_scores are too low
    
    It looks like swapping test and source, or flipping source
    gives different accuracy scores.
    OP1 and OP2 are slightly different
    SAE is more strict vs RMSE
    """
