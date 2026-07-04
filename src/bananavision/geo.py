from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .models import GeoPoint

GPS_IFD_TAG = 34853
GPS_LAT_REF = 1
GPS_LATITUDE = 2
GPS_LON_REF = 3
GPS_LONGITUDE = 4
GPS_ALT_REF = 5
GPS_ALTITUDE = 6


@dataclass(frozen=True)
class GeoTransform:
    origin_x: float
    origin_y: float
    pixel_width: float
    pixel_height: float
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    crs: str | None = None

    def pixel_to_world(self, x: float, y: float) -> GeoPoint:
        world_x = self.origin_x + (x * self.pixel_width) + (y * self.rotation_x)
        world_y = self.origin_y + (x * self.rotation_y) + (y * self.pixel_height)
        return GeoPoint(world_x, world_y, self.crs)


@dataclass(frozen=True)
class ExifGeoTag:
    latitude: float
    longitude: float
    altitude_m: float | None = None
    crs: str = "EPSG:4326"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_world_file(path: str | Path) -> GeoTransform:
    values = [float(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != 6:
        raise ValueError(f"World file must contain 6 numeric lines: {path}")
    pixel_width, rotation_y, rotation_x, pixel_height, origin_x, origin_y = values
    return GeoTransform(
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        rotation_x=rotation_x,
        rotation_y=rotation_y,
    )


def find_world_file(image_path: str | Path) -> Path | None:
    image_path = Path(image_path)
    candidates = [
        image_path.with_suffix(".wld"),
        image_path.with_suffix(".jgw"),
        image_path.with_suffix(".pgw"),
        image_path.with_suffix(".tfw"),
        image_path.with_suffix(image_path.suffix + "w"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_geotransform(image_path: str | Path) -> GeoTransform | None:
    image_path = Path(image_path)
    world_file = find_world_file(image_path)
    if world_file:
        return read_world_file(world_file)
    try:
        import rasterio  # type: ignore

        with rasterio.open(image_path) as dataset:
            transform = dataset.transform
            crs = str(dataset.crs) if dataset.crs else None
            # rasterio devuelve la transformada IDENTIDAD y crs=None para imagenes
            # SIN georreferencia real (p.ej. un JPEG normal con solo EXIF GPS); no la
            # trates como geotransform valido o enmascararia el GPS EXIF real.
            if crs is None and getattr(transform, "is_identity", False):
                return None
            return GeoTransform(
                origin_x=transform.c,
                origin_y=transform.f,
                pixel_width=transform.a,
                pixel_height=transform.e,
                rotation_x=transform.b,
                rotation_y=transform.d,
                crs=crs,
            )
    except Exception:
        return None


def read_exif_geotag(image_path: str | Path) -> ExifGeoTag | None:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            gps = exif.get_ifd(GPS_IFD_TAG) if hasattr(exif, "get_ifd") else {}
    except Exception:
        return None
    if not gps:
        return None
    try:
        latitude = _dms_to_decimal(gps.get(GPS_LATITUDE), gps.get(GPS_LAT_REF))
        longitude = _dms_to_decimal(gps.get(GPS_LONGITUDE), gps.get(GPS_LON_REF))
    except Exception:
        return None
    if latitude is None or longitude is None:
        return None
    altitude = _gps_altitude(gps.get(GPS_ALTITUDE), gps.get(GPS_ALT_REF))
    return ExifGeoTag(latitude=latitude, longitude=longitude, altitude_m=altitude)


def describe_image_georeference(image_path: str | Path) -> dict[str, Any] | None:
    if find_world_file(image_path):
        return {"type": "world_file"}
    transform = load_geotransform(image_path)
    if transform is not None:
        return {"type": "geotransform", "crs": transform.crs}
    geotag = read_exif_geotag(image_path)
    if geotag is None:
        return None
    return {"type": "exif_gps", "geotag": geotag.to_dict()}


def has_image_georeference(image_path: str | Path) -> bool:
    return describe_image_georeference(image_path) is not None


def _dms_to_decimal(value: Any, ref: Any) -> float | None:
    if value is None:
        return None
    parts = list(value)
    if len(parts) < 3:
        return None
    degrees = _rational_to_float(parts[0])
    minutes = _rational_to_float(parts[1])
    seconds = _rational_to_float(parts[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if _clean_ref(ref) in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _gps_altitude(value: Any, ref: Any) -> float | None:
    if value is None:
        return None
    altitude = _rational_to_float(value)
    if _altitude_ref(ref) == 1:
        altitude *= -1.0
    return altitude


def _rational_to_float(value: Any) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        denominator = float(value[1])
        return float(value[0]) / denominator if denominator else 0.0
    return float(value)


def _clean_ref(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore").strip("\x00").upper()
    return str(value).strip().upper()


def _altitude_ref(value: Any) -> int:
    if isinstance(value, bytes):
        return int(value[0]) if value else 0
    if value is None:
        return 0
    return int(value)
