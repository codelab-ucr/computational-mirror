import cv2
import numpy as np

def sift_reg(img1, img2, output_path='aligned_sift_result.jpg'):
    """
    Aligns grayscale image2 to match image1 using SIFT features.
    """
    
    if img1 is None or img2 is None:
        print("Error: Could not load images. Check file paths.")
        return

    # 2. Detect Features using SIFT
    # SIFT is robust against scale variations (zoom differences)
    sift = cv2.SIFT_create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 3. Match Features (Euclidean/L2 Distance)
    # We use knnMatch to get the 2 best matches for every feature
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 4. Apply Lowe's Ratio Test
    # This filters out ambiguous matches. We only keep a match if the 
    # closest match is significantly closer than the second closest match.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches using SIFT.")

    # Minimum number of matches needed to compute Homography is 4
    if len(good_matches) < 4:
        print("Error: Not enough good matches found to compute homography.")
        return

    # Visualize the filtered matches
    # flags=2 draws only the matches, not the surrounding keypoints
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    cv2.imwrite("debug_sift_matches.jpg", match_img)

    # 5. Extract Coordinates
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 6. Compute Homography
    # Note: We find the matrix from Image 2 (dst in list) to Image 1 (src in list)
    # The logic here is: warp(img2) -> img1 space
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 7. Warp Image
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
        best_score (float): The Laplacian variance score of the winner.
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
        
        # Ensure the slice is compatible with cv2.Laplacian
        # If the data is float (0.0-1.0), we might want to scale it or handle it carefully,
        # but cv2.Laplacian handles float inputs fine (returns float64).
        
        # Compute Sharpness (Variance of Laplacian)
        #score = cv2.Laplacian(current_slice, cv2.CV_64F).var()
        grad_x = cv2.Sobel(current_slice, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(current_slice, cv2.CV_64F, 0, 1, ksize=3)
        score = np.hypot(grad_x, grad_y).mean()
        
        if score > best_score:
            best_score = score
            best_index = i

    # 3. Retrieve the winner
    best_image = image_stack[best_index]
    
    return best_index, best_image, best_score