import cv2
import numpy as np

def sift_reg(img1, img2):
    """
    Aligns grayscale image2 to match image1 using SIFT features.

    Returns:
        M (np.ndarray): 3x3 homography from image2 to image1.
        aligned_img (np.ndarray): image2 warped into image1's frame.
    """

    # 1. Detect Features using SIFT
    # SIFT is robust against scale variations (zoom differences)
    sift = cv2.SIFT_create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 2. Match Features (Euclidean/L2 Distance)
    # We use knnMatch to get the 2 best matches for every feature
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 3. Apply Lowe's Ratio Test
    # This filters out ambiguous matches. We only keep a match if the 
    # closest match is significantly closer than the second closest match.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches using SIFT.")

    # Minimum number of matches needed to compute Homography is 4
    if len(good_matches) < 4:
        raise RuntimeError(f'Only {len(good_matches)} good SIFT matches: at least 4 are needed for a homography.')

    # 4. Extract Coordinates
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 5. Compute Homography
    # Note: We find the matrix from Image 2 (dst in list) to Image 1 (src in list)
    # The logic here is: warp(img2) -> img1 space
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 6. Warp Image
    h, w = img1.shape
    aligned_img = cv2.warpPerspective(img2, M, (w, h))

    return M, aligned_img

def find_sharpest_from_stack(image_stack):
    """
    Identifies the sharpest image slice within a 3D numpy array.
    
    Args:
        image_stack (np.ndarray): A 3D array of shape (N, Height, Width).
                                  N = number of images.
                                  Assumes data is uint8 or float.
    
    Returns:
        best_index (int): The index of the sharpest image in the stack.
        best_image (np.ndarray): The actual sharpest 2D image slice.
        best_score (float): The sharpness score (mean Sobel gradient magnitude) of the winner.
    """
    
    # 1. Validation
    if image_stack.ndim != 3:
        raise ValueError(f"Input must be a 3D array. Received shape: {image_stack.shape}")

    n_images = image_stack.shape[0]
    best_score = -1
    best_index = -1
    
    # 2. Iterate through the stack
    for i in range(n_images):
        # Extract the 2D slice
        current_slice = image_stack[i]

        # Sharpness: mean Sobel gradient magnitude
        grad_x = cv2.Sobel(current_slice, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(current_slice, cv2.CV_64F, 0, 1, ksize=3)
        score = np.hypot(grad_x, grad_y).mean()
        
        if score > best_score:
            best_score = score
            best_index = i

    # 3. Retrieve the winner
    best_image = image_stack[best_index]
    
    return best_index, best_image, best_score