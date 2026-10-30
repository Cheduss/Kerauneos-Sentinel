
from pathlib import Path

import cv2
import numpy as np

from app.optical_flow import estimate_motion
from app.motion_visualization import draw_motion_vectors

"""

ESTO ES UNA DEMO PARA VER SI FUNCIONA.-

"""


def main():

    # Imagen sintetica de prueba
    image1 = np.zeros((400, 500, 3), dtype=np.uint8)

    rng = np.random.default_rng(42)

    # Crear objetos con caracteristicas visuales
    for _ in range(35):
        x = int(rng.integers(30, 450))
        y = int(rng.integers(30, 350))
        radius = int(rng.integers(4, 12))

        cv2.circle(
            image1,
            (x, y),
            radius,
            (0, 200, 255),
            -1
        )

    # Desplazamiento artificial:
    # 8 pixeles a la derecha y 5 hacia arriba
    transformation = np.float32([
        [1, 0, 8],
        [0, 1, -5]
    ])

    image2 = cv2.warpAffine(
        image1,
        transformation,
        (500, 400)
    )

    # Calcular el movimiento
    motion = estimate_motion(image1, image2)

    # Dibujar las flechas
    result = draw_motion_vectors(
        image2,
        motion,
        max_vectors=60
    )

    # Guardar resultados
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(
        str(output_dir / "demo_t1.png"),
        image1
    )

    cv2.imwrite(
        str(output_dir / "demo_t2.png"),
        image2
    )

    cv2.imwrite(
        str(output_dir / "demo_movimiento.png"),
        result
    )

    print("\nKERAUNEOS-SENTINEL")
    print("Demostracion de flujo optico")
    print("-" * 30)

    print(
        f"Puntos seguidos: {motion['tracked_points']}"
    )

    print(
        f"DX: {motion['dx_pixels']} pixeles"
    )

    print(
        f"DY: {motion['dy_pixels']} pixeles"
    )

    print("\nResultado: outputs/demo_movimiento.png")


if __name__ == "__main__":
    main()
