import numpy as np
import cv2
from src.geometry.camera import create_intrinsic_matrix
from src.geometry.projection import project_points
from pathlib import Path

if __name__ == "__main__":
    height = 480
    width = 640
    channels = 3
    color = (0, 0, 255)
    radius = 5
    matrix = [800, 800, 320, 240]
    points = np.array([
        [0, 0, 5],
        [1, 0, 5],
        [-1, 0, 5],
        [0, 1, 5],
        [0, -1, 5],
        [5, 0, 5]
    ], dtype=np.float64)
    image = np.zeros((height, width, channels), dtype=np.uint8)

    output_dir = Path("C:/Users/admin/source/3D-Vision-Learning/results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = output_dir / "projected_points.png"

    camera_matrix = create_intrinsic_matrix(matrix)
    pixel_points, valid_mask = project_points(points, camera_matrix)

    visible_count = 0
    for index, pixel in enumerate(pixel_points):
        u, v = pixel
        u = int(u)
        v = int(v)
        label = "P" + str(index)

        if (0 <= u < width) and (0 <= v < height):
            cv2.circle(image, (u, v), radius, color, -1)
            cv2.putText(image, label, (u-8, v-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            visible_count += 1
    assert visible_count == 5

    print(f"pixel_points shape : {pixel_points.shape}")
    print(f"valid_mask : {valid_mask}")
    print(f"visible_count : {visible_count}")

    save_success = cv2.imwrite(file_name, image)
    print(f"저장 경로 : {file_name}")
    print(f"저장 성공 : {save_success}")

    cv2.imshow("Projected points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()