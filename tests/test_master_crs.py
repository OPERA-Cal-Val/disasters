import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from disasters.mosaic import _bounds_intersect, get_master_crs, utm_crs_for_bbox

BAY_AREA_AOI = [37.3, 37.9, -122.6, -121.8]  # [S, N, W, E], UTM zone 10N


def _write_tile(path, crs, origin_x, origin_y):
    """Write a small 30 m tile (3 km x 3 km) in the given CRS."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(origin_x, origin_y, 30, 30),
    ) as dst:
        dst.write(np.zeros((100, 100), dtype="uint8"), 1)
    return str(path)


def _local_df(paths):
    # Local mode: no "Geometry (WKT)" column, only file paths
    return pd.DataFrame({"Download URL WTR": paths})


def test_local_master_crs_uses_rasters_overlapping_the_aoi(tmp_path):
    # One zone-10 tile inside the AOI (near San Jose), three zone-11 tiles far away
    inside = _write_tile(tmp_path / "z10.tif", "EPSG:32610", 590000, 4140000)
    elsewhere = [
        _write_tile(tmp_path / f"z11_{i}.tif", "EPSG:32611", 400000 + i * 5000, 4000000)
        for i in range(3)
    ]

    crs = get_master_crs(_local_df([inside] + elsewhere), "flood", bbox=BAY_AREA_AOI)

    assert crs == "EPSG:32610"


def test_local_master_crs_without_overlap_uses_aoi_utm_zone(tmp_path):
    far = _write_tile(tmp_path / "z11.tif", "EPSG:32611", 400000, 4000000)

    assert get_master_crs(_local_df([far]), "flood", bbox=BAY_AREA_AOI) == "EPSG:32610"


def test_master_crs_without_bbox_keeps_most_common(tmp_path):
    paths = [_write_tile(tmp_path / "z10.tif", "EPSG:32610", 590000, 4140000)] + [
        _write_tile(tmp_path / f"z11_{i}.tif", "EPSG:32611", 400000, 4000000)
        for i in range(2)
    ]

    assert get_master_crs(_local_df(paths), "flood") == "EPSG:32611"


def test_utm_crs_for_bbox():
    assert utm_crs_for_bbox(BAY_AREA_AOI) == "EPSG:32610"
    assert utm_crs_for_bbox([-30.15, -29.85, -51.30, -51.10]) == "EPSG:32722"
    # Antimeridian-crossing box centered near 175°W -> zone 1
    assert utm_crs_for_bbox([50, 52, 175, -165]) == "EPSG:32601"


def test_bounds_intersect_handles_antimeridian():
    aoi = [50, 52, 175, -175]
    assert _bounds_intersect([50.5, 51, 178, 179], aoi)
    assert _bounds_intersect([50.5, 51, -179, -178], aoi)
    assert not _bounds_intersect([50.5, 51, 0, 1], aoi)
    assert not _bounds_intersect([10, 11, 178, 179], aoi)
