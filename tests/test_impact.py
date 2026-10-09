import logging
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from disasters.impact import compute_structure_impact

ORIGIN_X, ORIGIN_Y = 500000, 4000000


def _write_extent(path):
    """10x10 max-extent raster at 30 m: left half flooded (1), right half dry (0)."""
    data = np.zeros((10, 10), dtype="uint8")
    data[:, :5] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint8",
        crs="EPSG:32610",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, 30, 30),
        nodata=255,
    ) as dst:
        dst.write(data, 1)
    return path


def _buildings():
    """One building fully in the flooded half, one in the dry half (as OSM returns, EPSG:4326)."""
    wet = box(ORIGIN_X + 30, ORIGIN_Y - 90, ORIGIN_X + 90, ORIGIN_Y - 30)
    dry = box(ORIGIN_X + 210, ORIGIN_Y - 90, ORIGIN_X + 270, ORIGIN_Y - 30)
    gdf = gpd.GeoDataFrame(
        {"building": ["yes", "yes"]}, geometry=[wet, dry], crs="EPSG:32610"
    )
    return gdf.to_crs("EPSG:4326")


@patch("disasters.impact.ox.features_from_polygon")
def test_compute_structure_impact_returns_total_and_flooded(mock_osm, tmp_path):
    mock_osm.return_value = _buildings()
    raster = _write_extent(tmp_path / "max_extent.tif")

    total, flooded = compute_structure_impact(raster, None, tmp_path / "impact.csv")

    assert (total, flooded) == (2, 1)
    assert (tmp_path / "impact.csv").exists()


@patch("disasters.impact.ox.features_from_polygon")
def test_compute_structure_impact_warns_when_osm_fails(mock_osm, tmp_path, caplog):
    mock_osm.side_effect = TimeoutError("overpass timeout")
    raster = _write_extent(tmp_path / "max_extent.tif")

    with caplog.at_level(logging.WARNING, logger="disasters.impact"):
        total, flooded = compute_structure_impact(raster, None, tmp_path / "impact.csv")

    assert (total, flooded) == (0, 0)
    assert "INCOMPLETE" in caplog.text
