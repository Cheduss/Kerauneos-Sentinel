
import csv

from app.export_results import save_cells_csv


def test_export_csv(tmp_path):
    cells = [
        {
            "id": 1,
            "area": 250,
            "centroid": {"x": 15.5, "y": 20.5},
            "bounding_box": {
                "x": 5,
                "y": 10,
                "width": 20,
                "height": 25
            }
        }
    ]

    path = tmp_path / "nucleos.csv"

    result = save_cells_csv(cells, path)

    assert result.exists()

    with result.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["area_px"] == "250"
    assert rows[0]["centroid_x"] == "15.5"
    assert rows[0]["bbox_width"] == "20"


def test_export_empty_cells(tmp_path):
    path = tmp_path / "empty.csv"

    save_cells_csv([], path)

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == []


def test_export_creates_directory(tmp_path):
    path = tmp_path / "results" / "nucleos.csv"

    save_cells_csv([], path)

    assert path.is_file()
