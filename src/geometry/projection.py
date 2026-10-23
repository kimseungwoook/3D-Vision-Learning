import numpy as np
from src.geometry.camera import create_intrinsic_matrix

def project_points(points, K):
    """
    param points : 3D 좌표 (N, 3)
    param K : Intrinsic Matrix (3, 3)

    return : pixel_points, valid_mask
            pixel_points : 3D 좌표 -> 2D 좌표
            valid_mask : Z가 양수인 points만 저장  -> [True, False, True, ....]
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 배열은 (N, 3)형태를 만족해야 합니다.")

    if K.ndim != 2 or K.shape != (3, 3):
        raise ValueError("K 행렬은 (3, 3) 크기의 2차원 배열이어야 합니다.")

    z_values = points[:, 2]
    valid_mask = z_values > 0

    valid_points = points[valid_mask]
    # 동차 좌표
    homogeneous_points = valid_points @ K.T

    # 3D 좌표 -> 2D 좌표 변환
    XY = homogeneous_points[:, :2]
    Z = homogeneous_points[:, 2:3]

    pixel_points = XY / Z

    return pixel_points, valid_mask

if __name__ == "__main__":
    points = np.array([
        [2, 1, 4],
        [3, 2, -1],
        [1, 1, 0],
        [4, 3, 8]
    ], dtype=np.float64)
    matrix = [800, 800, 320, 240]

    camera_matrix = create_intrinsic_matrix(matrix)

    pixel_points, valid_mask = project_points(points, camera_matrix)

    print(f"pixel_points : \n{pixel_points}")
    print(f"pixel_points Shape : {pixel_points.shape}")
    print(f"\nvalid_mask : {valid_mask}")
    print(f"valid_mask Shape : {valid_mask.shape}")