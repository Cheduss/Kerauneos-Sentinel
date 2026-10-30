
import csv
from pathlib import Path

"""

ESTO SE UTILIZA PARA EXPORTAR A UN CSV LA INFORMACIÓN DE LO DETECTADO EN LA IMAGEN DE RADAR

"""


CSV_FIELDS = [
    "id",
    "area_px",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
]


def save_cells_csv(
    cells: list[dict],
    output_path: str | Path
) -> Path:
    """
    Exporta las caracteristicas de los nucleos
    detectados a un archivo CSV.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()

        for cell in cells:
            centroid = cell["centroid"]
            bbox = cell["bounding_box"]

            writer.writerow({
                "id": cell["id"],
                "area_px": cell["area"],
                "centroid_x": centroid["x"],
                "centroid_y": centroid["y"],
                "bbox_x": bbox["x"],
                "bbox_y": bbox["y"],
                "bbox_width": bbox["width"],
                "bbox_height": bbox["height"],
            })

    return path
