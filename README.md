# Kerauneos-Sentinel 🌩️

**Detección de núcleos de precipitación y estimación de desplazamientos aparentes en imágenes de radar meteorológico mediante procesamiento digital de imágenes.**

Proyecto desarrollado para la materia **Procesamiento de Imágenes**. Combina un backend en Python (OpenCV y FastAPI) con una interfaz web en HTML, CSS y JavaScript.

> **Alcance:** es una herramienta experimental y educativa. Trabaja con imágenes de radar; no reemplaza el procesamiento de volúmenes originales ni constituye un sistema operativo de alertas meteorológicas.

## Funcionalidades

### 1. Detección de núcleos

- Carga de capturas PNG/JPG desde el navegador.
- Filtrado espacial: ninguno, promedio, gaussiano o mediana.
- Segmentación por colores en HSV o segmentación aproximada por umbral de reflectividad (dBZ) a partir de la paleta de una captura compatible.
- Operaciones morfológicas: erosión, dilatación, apertura y cierre.
- Detección de componentes conectados y cálculo de área (px²), centroide y caja delimitadora (*bounding box*).
- Vistas **Original**, **Máscara binaria** y **Procesada**; exportación de imagen PNG y resultados CSV.

### 2. Seguimiento temporal

- Carga de dos imágenes (T1 y T2) con **las mismas dimensiones y el mismo encuadre**.
- Flujo óptico piramidal **Lucas–Kanade** para seguir características entre capturas.
- Dos modos: **general**, sobre la imagen; y **precipitación**, con selección inicial de puntos en regiones rojas y magentas (HSV), más comprobaciones de seguimiento.
- Flechas sobre la captura de T2, vistas T1/T2/Movimiento y exportación de imagen PNG y datos JSON.
- Tabla con Δx, Δy, **distancia en píxeles** y **dirección en grados** para las flechas mostradas.

La dirección indica **hacia dónde apunta el desplazamiento**, medida desde el norte en sentido horario (0° N, 90° E, 180° S, 270° O), **suponiendo que la imagen está orientada con el norte hacia arriba**.

**No se calculan velocidades en km/h:** las capturas pueden tener intervalos irregulares y no contamos todavía con una escala espacial validada para cada imagen.

## Instalación en Windows (PowerShell)

Se recomienda **Python 3.12**.

```powershell
# Desde la carpeta raíz de Kerauneos-Sentinel
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si ya existe y está activo tu entorno `.venv`, podés omitir su creación y activación.

### Iniciar la aplicación

```powershell
python -m uvicorn app.api:app --reload
```

Abrí en el navegador:

- **Aplicación:** http://127.0.0.1:8000/app
- **Documentación interactiva de la API:** http://127.0.0.1:8000/docs

Detené el servidor con `Ctrl + C`.

## Primer análisis recomendado

En la pestaña **Detección de núcleos**:

1. Cargá `imagen_test/radar_completo.png` (si está disponible en tu copia del repositorio).
2. Elegí segmentación **dBZ**, umbral **40 dBZ**, filtro **Ninguno**, morfología **Apertura** y área mínima **100 px²**.
3. Presioná **Ejecutar análisis** y alterná entre las tres vistas.
4. Revisá la tabla de núcleos y descargá el PNG o CSV.

En **Seguimiento temporal**, cargá dos capturas consecutivas del mismo radar, elegí el modo **Precipitación** y presioná **Estimar desplazamiento**. La tabla informará los vectores detectados y sus direcciones; también podés descargar los resultados en JSON.

## Recorte interno y compatibilidad de capturas

Para el **perfil de captura SINARAME implementado y reconocido** (dimensiones **805 × 802 píxeles**, con ubicación y colores de leyenda compatibles con la referencia), el backend procesa automáticamente una región útil y devuelve las anotaciones sobre la **imagen completa**, conservando su contexto geográfico.

El perfil se desarrolló a partir de una captura del radar de Ezeiza. **No se garantiza todavía que todas las imágenes de SINARAME sean reconocidas**: otras resoluciones, recortes, paletas o cambios de interfaz pueden requerir adaptar el perfil. Cuando no se reconoce un perfil compatible, el modo HSV puede trabajar con la imagen completa; el modo dBZ exige la validación de su paleta de referencia.

La API conserva un parámetro `roi` opcional para usos avanzados; la interfaz principal no requiere introducir coordenadas.

## Estructura del proyecto

```text
Kerauneos-Sentinel/
├── app/                      # Procesamiento y API FastAPI
│   ├── api.py
│   ├── pipeline.py
│   ├── segmentation.py
│   ├── reflectivity.py
│   ├── morphology.py
│   ├── detection.py
│   ├── optical_flow.py
│   └── storm_tracking.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── tracking.js
├── imagen_test/              # Capturas utilizadas para las pruebas
├── tests/                    # Pruebas automatizadas
├── run_analysis.py           # Alternativa por consola para detección
├── run_real_tracking.py      # Demostración por consola de seguimiento
├── requirements.txt
└── README.md
```

El backend también incluye módulos de carga, filtrado, recorte, visualización y exportación de datos.

## Uso desde la terminal (opcional)

Detección:

```powershell
python run_analysis.py imagen_test/radar_completo.png --segmentation dbz --dbz 40 --morphology opening --min-area 100
```

Seguimiento por consola (dos archivos del mismo encuadre):

```powershell
python run_real_tracking.py imagen_test/seguimiento/radar_t1.png imagen_test/seguimiento/radar_t2.png --roi 8 92 659 739
```

> El script `run_real_tracking.py` es una herramienta independiente de demostración; a diferencia de la API web, su recorte es manual y sus resultados se guardan en `outputs/real_tracking/`.

## Pruebas automatizadas

```powershell
python -m pytest -v
```

Las pruebas cubren los módulos de procesamiento, las rutas de FastAPI y aspectos de integración del frontend.

## Tecnologías

**Python**, **OpenCV**, **NumPy**, **Pillow**, **FastAPI**, **Uvicorn**, **pytest**, **HTML**, **CSS** y **JavaScript**.

Este proyecto **no entrena ni utiliza modelos de inteligencia artificial o aprendizaje automático**. Lucas–Kanade es un algoritmo clásico de flujo óptico, no una red neuronal entrenada.

## Limitaciones conocidas

- La reflectividad en dBZ se **estima a partir de los colores de la captura**; no equivale a leer valores originales calibrados del radar.
- Los rótulos, límites y demás anotaciones de la imagen pueden causar falsas detecciones o afectar el seguimiento.
- Las flechas siguen **características visuales**, no identifican necesariamente la trayectoria de una celda convectiva completa.
- El vector mediano resume varios desplazamientos y puede ocultar movimientos distintos dentro de un sistema.
- Las distancias están expresadas en **píxeles**, no en kilómetros; no se generan velocidades meteorológicas sin tiempo y escala validados.
- La compatibilidad de perfiles SINARAME debe verificarse con imágenes de distintos radares y formatos.

## Posibles mejoras futuras

- Lectura y validación automática de fechas/horas de captura, sin suponer una cadencia fija.
- Georreferenciación y cálculo de desplazamientos físicos cuando se disponga de escala espacial confiable.
- Seguimiento por regiones o núcleos individuales, además del seguimiento de puntos.
- Validación con más radares SINARAME y formatos de imagen.
- Publicación web y evaluación del despliegue en Vercel u otro servicio.

---

**Kerauneos-Sentinel** · Proyecto de Procesamiento de Imágenes.
