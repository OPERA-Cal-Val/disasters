"""Tests that auxiliary layers (CONF, VEG-DIST-DATE, VEG-DIST-CONF) stay paired with their granules."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray
from rasterio.transform import from_origin

from disasters.filters import (
    filter_by_date_and_confidence,
    reclassify_snow_ice_as_water,
)
from disasters.mosaic import compile_and_load_data, mosaic_opera
from disasters.pipeline import aligned_layer_links, stack_with_conf


def _write_tif(path, value, nodata=255, shape=(4, 4)):
    """Write a small uint8 raster filled with `value` (or using an explicit array)."""
    data = value if isinstance(value, np.ndarray) else np.full(shape, value, "uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32610",
        transform=from_origin(500000, 4000000, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _load(path):
    da = rioxarray.open_rasterio(path, masked=False)
    da.load()
    return da


def test_aligned_layer_links_keeps_missing_aux_in_place():
    df = pd.DataFrame(
        {
            "Download URL WTR": ["a_WTR", "b_WTR", None, "d_WTR"],
            "Download URL CONF": ["a_CONF", None, "c_CONF", "d_CONF"],
        }
    )

    urls, aux = aligned_layer_links(
        df, "Download URL WTR", ["Download URL CONF", "Download URL MISSING"]
    )

    # Row without a primary URL is dropped; CONF stays paired with its own granule
    assert urls == ["a_WTR", "b_WTR", "d_WTR"]
    assert aux["Download URL CONF"] == ["a_CONF", None, "d_CONF"]
    assert aux["Download URL MISSING"] is None


def test_compile_and_load_data_returns_aligned_aux(tmp_path):
    wtr = [_write_tif(tmp_path / f"g{i}_WTR.tif", i) for i in range(3)]
    conf = [
        _write_tif(tmp_path / "g0_CONF.tif", 10),
        None,
        _write_tif(tmp_path / "g2_CONF.tif", 12),
    ]

    DS, conf_DS, date_DS = compile_and_load_data(wtr, conf_layer_links=conf)

    assert len(DS) == len(conf_DS) == 3
    assert conf_DS[1] is None
    assert int(conf_DS[0].values.max()) == 10
    assert int(conf_DS[2].values.max()) == 12
    assert date_DS is None


def test_compile_and_load_data_all_missing_aux_returns_none(tmp_path):
    wtr = [_write_tif(tmp_path / f"g{i}_WTR.tif", i) for i in range(3)]

    DS, conf_DS, date_DS = compile_and_load_data(
        wtr, conf_layer_links=[None, None, None], date_layer_links=[None] * 3
    )

    # Previously, 3 granules with no aux layers could be unpacked one-per-name
    assert len(DS) == 3
    assert conf_DS is None
    assert date_DS is None


def test_compile_and_load_data_rejects_misaligned_aux(tmp_path):
    wtr = [_write_tif(tmp_path / f"g{i}_WTR.tif", i) for i in range(2)]
    with pytest.raises(ValueError, match="aligned"):
        compile_and_load_data(wtr, conf_layer_links=[wtr[0]])


def test_satellite_filter_applies_to_aux_consistently(tmp_path):
    names = ["x_S1A_1", "x_S1C_2", "x_S1A_3"]
    wtr = [_write_tif(tmp_path / f"{n}_WTR.tif", i) for i, n in enumerate(names)]
    conf = [_write_tif(tmp_path / f"{n}_CONF.tif", 10 + i) for i, n in enumerate(names)]

    DS, conf_DS, _ = compile_and_load_data(wtr, conf_layer_links=conf)

    # S1A is most common; the S1C granule and *its* CONF are dropped together
    assert [int(d.values.max()) for d in DS] == [0, 2]
    assert [int(c.values.max()) for c in conf_DS] == [10, 12]


def test_dist_filter_drops_granule_without_date_when_filter_date_set(tmp_path):
    status = [_load(_write_tif(tmp_path / f"s{i}.tif", 6)) for i in range(2)]
    dates = [_load(_write_tif(tmp_path / "d0.tif", 200)), None]

    filtered, _ = filter_by_date_and_confidence(
        status, dates, date_threshold=100, drop_missing_dates=True
    )

    assert len(filtered) == 1


def test_dist_filter_keeps_granule_without_date_when_no_filter_date(tmp_path):
    data = np.array([[0, 6], [6, 0]], dtype="uint8")
    status = [_load(_write_tif(tmp_path / "s0.tif", data))]

    filtered, _ = filter_by_date_and_confidence(
        status, [None], date_threshold=0, drop_missing_dates=False
    )

    assert len(filtered) == 1
    out = filtered[0].squeeze().values
    # "No disturbance" (0) is masked to nodata as for other granules; disturbance is kept
    assert out[0, 1] == 6 and out[0, 0] == 255


def test_reclassify_snow_ice_skips_granule_without_conf(tmp_path):
    wtr = [_load(_write_tif(tmp_path / f"w{i}.tif", 252)) for i in range(2)]
    conf = [_load(_write_tif(tmp_path / "c0.tif", 1)), None]

    out, _ = reclassify_snow_ice_as_water(wtr, conf)

    assert int(out[0].values.max()) == 1  # reclassified via CONF
    assert int(out[1].values.max()) == 252  # untouched: no CONF available


def test_stacked_conf_follows_winning_granule_with_missing_conf(tmp_path):
    # Granule A: not water (0) with no CONF. Granule B: open water (1) with CONF=1.
    wtr_a = _load(_write_tif(tmp_path / "a_WTR.tif", 0))
    wtr_b = _load(_write_tif(tmp_path / "b_WTR.tif", 1))
    conf_b = _load(_write_tif(tmp_path / "b_CONF.tif", 1))

    stacked = [stack_with_conf(wtr_a, None), stack_with_conf(wtr_b, conf_b)]
    assert all(s.shape[0] == 2 for s in stacked)
    assert int(stacked[0].isel(band=1).values.min()) == 255

    mosaic, _, _ = mosaic_opera(stacked, product="OPERA_L3_DSWX-S1_V1", layer="WTR")

    # Water outranks not-water, and the CONF band comes from the same (winning) granule
    assert int(mosaic.isel(band=0).values.min()) == 1
    assert int(mosaic.isel(band=1).values.min()) == 1
