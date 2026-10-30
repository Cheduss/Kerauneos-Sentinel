
from pathlib import Path

import cv2
import numpy as np

from app.storm_tracking import estimate_storm_motion
from app.motion_visualization import draw_motion_vectors


#ESTO ES PARA VER SI FUNCIONA.

def main():
    first = np.zeros((240, 300, 3), dtype=np.uint8)

    # Crear ecos de precipitacion sinteticos
    for x in range(25, 210, 45):
        for y in range(35, 190, 55):
            cv2.rectangle(
                first, (x, y), (x + 19, y + 19),
                (0, 0, 255), -1
            )
            cv2.rectangle(
                first, (x + 4, y + 4),
                (x + 10, y + 10),
                (255, 0, 255), -1
            )

    # Mover los ecos 8 px a la derecha
    # y 5 px hacia arriba
    matrix = np.float32([
        [1, 0, 8],
        [0, 1, -5]
    ])

    second = cv2.warpAffine(
        first, matrix, (300, 240)
    )

    motion = estimate_storm_motion(first, second)

    result = draw_motion_vectors(
        second, motion, max_vectors=30
    )

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    cv2.imwrite(str(output / "storm_t1.png"), first)
    cv2.imwrite(str(output / "storm_t2.png"), second)
    cv2.imwrite(str(output / "storm_motion.png"), result)

    print("KERAUNEOS-SENTINEL")
    print(f"Puntos candidatos: {motion['candidate_points']}")
    print(f"Puntos validos: {motion['tracked_points']}")
    print(f"DX: {motion['dx_pixels']} px")
    print(f"DY: {motion['dy_pixels']} px")
    print("Resultado: outputs/storm_motion.png")


if __name__ == "__main__":
    main()
