import numpy as np

def create_intrinsic_matrix(matrix):
    fx = matrix[0]
    fy = matrix[1]
    cx = matrix[2]
    cy = matrix[3]

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float64)

    return camera_matrix

if __name__ == "__main__":
    matrix = [800, 800, 320, 240]
    camera_matrix = create_intrinsic_matrix(matrix)

    print(f"Camera Matrix : {camera_matrix}")
    print(f"Camera Matrix Shape : {camera_matrix.shape}")
    print(f"Camera Matrix dtype : {camera_matrix.dtype}")