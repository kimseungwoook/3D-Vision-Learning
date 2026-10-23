import pytest
import numpy as np
from src.geometry.camera import create_intrinsic_matrix
from src.geometry.projection import project_points

#
def test_optical_axis_projection():
    points = np.array([[0, 0, 5]], dtype=np.float64)
    matrix = [800, 800, 320, 240]

    camera_matrix = create_intrinsic_matrix(matrix)
    pixel_points, valid_mask = project_points(points, camera_matrix)

    np.testing.assert_array_equal(
        valid_mask,
        np.array([True])
    )

    np.testing.assert_allclose(
        pixel_points,
        np.array([[320, 240]])
    )

# Z 길이가 멀어질수록 3D 점이 이미지의 주점에 가까워지는지 확인
def test_depth_projection():
    points = np.array([[2, 0, 4],
                       [2, 0, 8]])
    matrix = [800, 800, 320, 240]

    camera_matrix = create_intrinsic_matrix(matrix)
    pixel_points, valid_mask = project_points(points, camera_matrix)

    np.testing.assert_array_equal(
        valid_mask,
        np.array([True, True])
    )

    np.testing.assert_allclose(
        pixel_points,
        np.array([[720, 240],
                  [520, 240]])
    )

# 유효하지 않은 Z <= 0 처리 검증
def test_invalid_depth_projection():
    points = np.array([
        [2, 1, 4],
        [3, 2, 0],
        [4, 3, -2]
    ], dtype=np.float64)
    matrix = [800, 800, 320, 240]

    camera_matrix = create_intrinsic_matrix(matrix)
    pixel_points, valid_mask = project_points(points, camera_matrix)

    np.testing.assert_array_equal(
        valid_mask,
        np.array([True, False, False])
    )

    np.testing.assert_allclose(
        pixel_points,
        np.array([[720, 440]])
    )

# 모드 3D 점이 유효하지 않을 때의 처리 검증
def test_all_invalid_depth():
    points = np.array([
        [2, 1, 0],
        [3, 2, -1],
        [4, 3, -5]
    ])
    matrix = [800, 800, 320, 240]

    camera_matrix = create_intrinsic_matrix(matrix)
    pixel_points, valid_mask = project_points(points, camera_matrix)

    np.testing.assert_array_equal(
        valid_mask,
        np.array([False, False, False])
    )

    assert pixel_points.shape == (0, 2)

# points shape 크기가 맞지 않을 때
def test_invalid_points_shape():
    points = np.array([1, 2, 3])
    matrix = [800, 800, 320, 240]

    camera_matrix = create_intrinsic_matrix(matrix)

    with pytest.raises(ValueError):
        project_points(points, camera_matrix)

# 카메라 내부 파라미터 행렬이 (3, 3)아닐 때 처리 검증
def test_invalid_intrinsic_shape():
    points = np.array([
        [2, 1, 4]
    ], dtype=np.float64)
    matrix = np.ones((4, 3))

    with pytest.raises(ValueError):
        project_points(points, matrix)

# 초점거리 변화 테스트
def test_focal_length_change():
    points = np.array([
        [1, 0, 5]
    ], dtype=np.float64)

    f_list = [800, 1200]
    camera_matrices = []

    for fx in f_list:
        matrix = [fx, 800, 320, 240]
        camera_matrix = create_intrinsic_matrix(matrix)
        camera_matrices.append(camera_matrix)

    camera_matrix1 = camera_matrices[0]
    camera_matrix2 = camera_matrices[1]

    pixel_point1, valid_mask1 = project_points(points, camera_matrix1)
    pixel_point2, valid_mask2 = project_points(points, camera_matrix2)

    np.testing.assert_allclose(
        pixel_point1,
        np.array([
            [480, 240]
        ])
    )
    np.testing.assert_allclose(
        pixel_point2,
        np.array([
            [560, 240]
        ])
    )
    np.testing.assert_array_equal(
        valid_mask1,
        np.array([True])
    )
    np.testing.assert_array_equal(
        valid_mask2,
        np.array([True])
    )