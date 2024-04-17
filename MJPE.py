import numpy as np


def MJPE(source_norm, test_norm):
    """
    Mean Joint Positional Error
    """
    # Calculate Euclidean distances for each joint pair and compute the mean.
    distances = np.linalg.norm(source_norm - test_norm, axis=1)
    mjpe = np.mean(distances)
    # return mjpe

    # Use max error to calculate accuracy
    max_acceptable_error = .61
    if mjpe >= max_acceptable_error:
        print("greater mjpe detected: ", mjpe)
        return 0 #100% error
    else:
        return 100 * (1 - (mjpe / max_acceptable_error))



def normalize_pose(pose3d):
    """
    Normalize the pose so that it centers within a sphere of radius 1.
    """
    # Centering the pose around the origin
    centroid = np.mean(pose3d, axis=0)
    centered_pose = pose3d - centroid
    
    # Scaling factor to fit the pose within a sphere of radius 1
    max_distance = np.max(np.linalg.norm(centered_pose, axis=1))
    scale_factor = 1 / max_distance
    
    # Applying the scaling
    normalized_pose = centered_pose * scale_factor
    return normalized_pose


def align_poses(A, B):
    """
    Rotates B to A's orientation
    """
    # Find optimal rotation through SVD.
    # Perform SVD to find the optimal rotation matrix R
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Ensure that the pose does not flip
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Rotate B
    rotatedB = np.dot(B, R)
    return rotatedB


def normalize_align_and_mjpe(source, test):
    # Normalize poses
    source_norm = normalize_pose(source)
    test_norm = normalize_pose(test)
    # Align test to source's orientation
    rotated_test = align_poses(source_norm, test_norm) 
    # Calculate and return MJPE
    return MJPE(source_norm, rotated_test)

# print(normalize_align_and_mjpe(pose1, pose2))


#Instructions:
    #1. Use MeTrabs to process the keypoints of source_pose and test_pose
    #2. pass the keypoints into normalize_align_and_mjpe()
    #3. return the MJPE score in a json.