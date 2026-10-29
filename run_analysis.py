
import argparse
from pathlib import Path

import cv2

from app.pipeline import analyze_radar


"""
ESTO ES TEMPORAL, HASTA QUE TENGAMOS EL FRONTEND (era para ver como funcionaba)

"""

def main():
    parser = argparse.ArgumentParser(
        description="Kerauneos-Sentinel - Analisis de radar"
    )

    parser.add_argument(
        "image",
        help="Ruta de la imagen de radar"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Area minima en pixeles"
    )

    parser.add_argument(
        "--filter",
        choices=["none", "average", "gaussian", "median"],
        default="none"
    )

    args = parser.parse_args()

    analysis = analyze_radar(
        image_path=args.image,
        min_area=args.min_area,
        filter_type=args.filter
    )

    # Crear carpeta de resultados
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    images = {
        "original.png": analysis["original"],
        "mascara.png": analysis["mask"],
        "resultado.png": analysis["result"]
    }

    for filename, image in images.items():
        success = cv2.imwrite(
            str(output_dir / filename),
            image
        )
        if not success:
            raise RuntimeError(
                f"No se pudo guardar {filename}"
            )

    print("\nKERAUNEOS-SENTINEL")
    print("-" * 30)
    print(f"Nucleos detectados: {analysis['total_cells']}")

    for cell in analysis["cells"]:
        print(f"\nNucleo {cell['id']}")
        print(f"Area: {cell['area']} pixeles")

        centroid = cell["centroid"]
        print(
            f"Centroide: "
            f"({centroid['x']}, {centroid['y']})"
        )

    print(f"\nResultados guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
