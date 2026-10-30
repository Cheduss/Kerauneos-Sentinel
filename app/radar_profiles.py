
import numpy as np

"""
UTILIZA LA BARRA DE MARCADORES DE LA IMAGEN D ERADAR QUE SUBI. Si los colores no coinciden
me da error. Es una certificacfion mas que una comprobacion de compatibilidad 8opcional)

"""

# Perfil de referencia para la imagen Ezeiza-SINARAME
EZEIZA_PROFILE = {
    "name": "Ezeiza-SINARAME",
    "width": 805,
    "height": 802,
    "legend_x": 730,
    "legend_y_top": 102,
    "legend_y_bottom": 714,
    "dbz_min": -15,
    "dbz_max": 75,
}

# Muestras BGR obtenidas de la barra de colores
# de la imagen de referencia.
REFERENCE_COLORS = {
    55: (205, 0, 203),
    45: (27, 0, 203),
    35: (48, 210, 205),
    20: (55, 242, 83),
}


def validate_ezeiza_radar(image: np.ndarray) -> bool:
    """
    Comprueba si una imagen tiene dimensiones y colores
    de leyenda compatibles con el perfil de referencia.

    No verifica el origen meteorologico de la imagen.
    """

    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise ValueError("Imagen BGR no valida.")

    height, width = image.shape[:2]

    if (
        width != EZEIZA_PROFILE["width"]
        or height != EZEIZA_PROFILE["height"]
    ):
        raise ValueError(
            "Dimensiones incompatibles con el perfil "
            "Ezeiza-SINARAME."
        )

    x = EZEIZA_PROFILE["legend_x"]
    top = EZEIZA_PROFILE["legend_y_top"]
    bottom = EZEIZA_PROFILE["legend_y_bottom"]

    dbz_max = EZEIZA_PROFILE["dbz_max"]
    dbz_min = EZEIZA_PROFILE["dbz_min"]

    for dbz, expected_color in REFERENCE_COLORS.items():

        # Posicion vertical esperada dentro de la escala
        fraction = (dbz_max - dbz) / (dbz_max - dbz_min)

        y = round(top + fraction * (bottom - top))

        actual = image[y, x].astype(np.int16)
        expected = np.array(
            expected_color, dtype=np.int16
        )

        # Tolerancia para pequenas diferencias de color
        difference = np.max(np.abs(actual - expected))

        if difference > 45:
            raise ValueError(
                "La escala de colores no es compatible "
                "con el perfil Ezeiza-SINARAME."
            )

    return True
