
import argparse
import json
from pathlib import Path

import cv2

from app.load_an_image import load_image
from app.crop_region import crop_radar
from app.storm_tracking import estimate_storm_motion
from app.motion_visualization import draw_motion_vectors


"""

PRUEBA PARA VER SI FUNCIONA

"""

def main():
    parser = argparse.ArgumentParser(
        description="Seguimiento temporal de ecos de radar"
    )

    parser.add_argument("image1", type=str)
    parser.add_argument("image2", type=str)

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None
    )

    parser.add_argument(
        "--max-corners",
        type=int,
        default=300
    )

    parser.add_argument(
        "--max-vectors",
        type=int,
        default=60
    )

    args = parser.parse_args()

    # 1. Cargar las dos imagenes
    first = load_image(args.image1)
    second = load_image(args.image2)

    if first.shape != second.shape:
        raise ValueError(
            "Las dos imagenes deben tener iguales dimensiones."
        )

    # 2. Recortar la misma region
    if args.roi is not None:
        first = crop_radar(first, *args.roi)
        second = crop_radar(second, *args.roi)

    # 3. Calcular movimiento de los ecos
    motion = estimate_storm_motion(
        first,
        second,
        max_corners=args.max_corners
    )

    # 4. Dibujar los vectores
    result = draw_motion_vectors(
        second,
        motion,
        max_vectors=args.max_vectors
    )

    # 5. Crear carpeta de resultados
    output_dir = Path("outputs/real_tracking")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / "radar_t1.png"),
        first
    )

    cv2.imwrite(
        str(output_dir / "radar_t2.png"),
        second
    )

    cv2.imwrite(
        str(output_dir / "movimiento.png"),
        result
    )

    # 6. Guardar resultados numericos
    summary = {
        "image1": args.image1,
        "image2": args.image2,
        "candidate_points": motion["candidate_points"],
        "tracked_points": motion["tracked_points"],
        "dx_pixels": motion["dx_pixels"],
        "dy_pixels": motion["dy_pixels"],
        "vectors": motion["vectors"]
    }

    with open(
        output_dir / "movimiento.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # 7. Mostrar resumen
    print("\nKERAUNEOS-SENTINEL")
    print("Seguimiento de ecos meteorologicos")
    print("-" * 40)

    print(
        f"Puntos candidatos: {motion['candidate_points']}"
    )

    print(
        f"Puntos validos: {motion['tracked_points']}"
    )

    print(
        f"DX: {motion['dx_pixels']:+.2f} px"
    )

    print(
        f"DY: {motion['dy_pixels']:+.2f} px"
    )

    print(
        f"\nResultados guardados en: {output_dir}"
    )

    if motion["tracked_points"] == 0:
        print(
            "Advertencia: no hay suficientes puntos "
            "validos para estimar el desplazamiento."
        )


if __name__ == "__main__":
    main()
