# Datos reales y entrenamiento del modelo de banano

Este directorio contiene el flujo para (re)entrenar el modelo con **imágenes UAV reales**.
El modelo incluido (`models/banana_real_v1.pt`) se entrenó con este proceso.

## Dataset real usado

**DS-v1 de AI-BananaMapping** — imágenes UAV RGB reales (nadir, baja altitud) de cultivos de
banano, tileadas a 1024 px, con anotaciones YOLO por planta.
- Fuente: Zenodo [record 20945958](https://zenodo.org/records/20945958), **licencia CC-BY-4.0**.
- ~14 180 train / 4 645 val / 4 611 test. Clase única: banano.

### Descargar y preparar

```bash
# 1. Descargar (9.8 GB, sin login)
curl -L -o ds-v1.rar "https://zenodo.org/api/records/20945958/files/ds-v1.rar/content"
7z x ds-v1.rar          # necesita 7-Zip / unrar

# 2. (opcional) colapsar clases a "banano" si tu dataset trae varias clases de enfermedad
python real_data/prepare_real_dataset.py --root DS-v1/ds-v1 --out DS-v1/banano_yolo
```

El `data.yaml` debe apuntar (ruta absoluta) a `images/{train,val,test}` con `names: [banano]`.

## Entrenar

```bash
pip install -e ".[ml]"
# YOLOv8 detección desde pesos COCO, sobre las imágenes reales:
yolo detect train model=yolov8s.pt data=DS-v1/ds-v1/banano_real.yaml epochs=25 imgsz=640 batch=24
```

> ⚠️ **En Windows**, entrena desde un archivo `.py` o el CLI `yolo` (NO `python - <<PY`
> por stdin) o el multiprocessing del DataLoader falla con `OSError: '<stdin>'`.

## Evaluar sobre el test real y registrar

```bash
python real_data/eval_real.py --weights runs/detect/train/weights/best.pt \
    --data DS-v1/ds-v1/banano_real.yaml --out real_eval
bananavision register-model runs/detect/train/weights/best.pt "real-v1.1" \
    --config configs/banana_real_model.yaml --acceptance-report real_eval/real_test_metrics.json
```

## Escala y afinado

El dataset es de **baja altitud (escala de planta)**. Para ortomosaicos de gran altura,
recorta tiles a tu GSD y **afina** (`resume`/transfer learning) con unas imágenes de tu
finca. El framework incluye split *group-aware* (sin fuga finca/bloque/vuelo),
`bananavision audit-dataset`, calibración y colas de *active learning* para acelerarlo.
