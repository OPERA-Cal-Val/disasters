from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin

from disasters.cli import cli
from disasters.filters import apply_slope_mask_to_raster, process_dem_and_slope
from disasters.io import parse_bbox_input
from disasters.pipeline import (
    PipelineConfig,
    run_download_only,
    run_pipeline,
    run_slope_filter_only,
)


# ---------------------------------------------------------
# 1. Test PipelineConfig handles lists correctly
# ---------------------------------------------------------
def test_pipeline_config_list_support():
    """Ensure the dataclass correctly accepts and stores our new list inputs."""
    config = PipelineConfig(
        bbox=[30.2, 30.3, -97.8, -97.7],
        output_dir=Path("/tmp/out"),
        layout_title="Test Layout",
        product=["OPERA_L3_DSWX-HLS_V1", "OPERA_L2_RTC-S1_V1"],
        satellites=["sentinel-1", "landsat"],
        search_dir=Path("/tmp/cached_search"),
    )

    assert isinstance(config.product, list)
    assert len(config.product) == 2
    assert "OPERA_L2_RTC-S1_V1" in config.product

    assert isinstance(config.satellites, list)
    assert "landsat" in config.satellites

    assert isinstance(config.search_dir, Path)


# ---------------------------------------------------------
# 2. Test run_pipeline sub-directory routing
# ---------------------------------------------------------
@patch("disasters.pipeline.authenticate", return_value=("user", "pass"))
@patch("disasters.pipeline.next_pass.run_next_pass")
@patch("disasters.pipeline.read_opera_metadata")
@patch("disasters.pipeline.generate_products")
def test_run_pipeline_multi_product_routing(
    mock_generate, mock_read_meta, mock_np, mock_auth, tmp_path
):
    """Ensure run_pipeline correctly loops through a list of products and creates dedicated subdirectories."""
    mock_read_meta.return_value = pd.DataFrame(
        {"Dataset": ["OPERA_L3_DSWX-HLS_V1", "OPERA_L2_RTC-S1_V1"]}
    )

    # Create a physical fake output directory so os.rename() doesn't fail with FileNotFoundError
    fake_np_out = tmp_path / "np_out"
    fake_np_out.mkdir()
    mock_np.return_value = str(fake_np_out)

    out_dir = tmp_path / "disasters_outputs"
    config = PipelineConfig(
        bbox=[30.2, 30.3, -97.8, -97.7],
        output_dir=out_dir,
        layout_title="Test",
        product=["OPERA_L3_DSWX-HLS_V1", "OPERA_L2_RTC-S1_V1"],
    )

    result_path = run_pipeline(config)

    assert result_path == out_dir
    assert mock_generate.call_count == 2

    call_1_kwargs = mock_generate.call_args_list[0].kwargs
    call_2_kwargs = mock_generate.call_args_list[1].kwargs

    assert call_1_kwargs["mode_dir"].name == "OPERA_L3_DSWX-HLS_V1"
    assert call_1_kwargs["product"] == "OPERA_L3_DSWX-HLS_V1"

    assert call_2_kwargs["mode_dir"].name == "OPERA_L2_RTC-S1_V1"
    assert call_2_kwargs["product"] == "OPERA_L2_RTC-S1_V1"


# ---------------------------------------------------------
# 3. Test run_download_only layer accumulation
# ---------------------------------------------------------
@patch("disasters.pipeline.authenticate", return_value=("user", "pass"))
@patch("disasters.pipeline.next_pass.run_next_pass")
@patch("disasters.pipeline.read_opera_metadata")
@patch("disasters.pipeline.concurrent.futures.ThreadPoolExecutor")
def test_run_download_only_accumulates_layers(
    mock_executor, mock_read_meta, mock_np, mock_auth, tmp_path
):
    """Ensure that passing multiple products extends the target_layers list correctly so nothing is skipped."""
    df = pd.DataFrame(
        {
            "Dataset": ["OPERA_L3_DSWX-HLS_V1", "OPERA_L2_RTC-S1_V1"],
            "Download URL WTR": ["https://fake.url/wtr.tif", None],
            "Download URL RTC-VV": [None, "https://fake.url/rtc.tif"],
        }
    )
    mock_read_meta.return_value = df

    # Create fake directory again
    fake_np_out = tmp_path / "np_out"
    fake_np_out.mkdir()
    mock_np.return_value = str(fake_np_out)

    out_dir = tmp_path / "downloads"

    result = run_download_only(
        bbox=[30.2, 30.3, -97.8, -97.7],
        output_dir=out_dir,
        product=["OPERA_L3_DSWX-HLS_V1", "OPERA_L2_RTC-S1_V1"],
    )

    assert result == out_dir / "data"


# ---------------------------------------------------------
# 4. Test CLI properly parses multiple flags into tuples
# ---------------------------------------------------------
def test_cli_multiple_flags():
    """Ensure Click bundles multiple -p and -s flags into tuples and passes them to the functions."""
    runner = CliRunner()

    # Mock it where it lives in pipeline, not where it is imported in cli
    with patch("disasters.pipeline.run_search_only") as mock_search:
        mock_search.return_value = Path("/tmp/fake_out")

        result = runner.invoke(
            cli,
            [
                "search",
                "-b",
                "30.2 30.3 -97.8 -97.7",
                "-o",
                "/tmp/fake_out",
                "-p",
                "OPERA_L3_DSWX-HLS_V1",
                "-p",
                "OPERA_L2_RTC-S1_V1",
                "-s",
                "sentinel-1",
                "-s",
                "landsat",
            ],
        )

        # Print output on failure to see exactly what click complained about
        if result.exit_code != 0:
            print(result.output)

        assert result.exit_code == 0

        # Verify the variables were unpacked properly
        kwargs = mock_search.call_args[1]
        assert kwargs["product"] == ["OPERA_L3_DSWX-HLS_V1", "OPERA_L2_RTC-S1_V1"]
        assert kwargs["satellites"] == ["sentinel-1", "landsat"]


def test_run_cli_parses_zoom_bbox_as_float_list(tmp_path):
    runner = CliRunner()

    with patch("disasters.cli.run_pipeline") as mock_run:
        mock_run.return_value = tmp_path / "flood"

        result = runner.invoke(
            cli,
            [
                "run",
                "-b",
                "30.2 30.3 -97.8 -97.7",
                "-zb",
                "30.21,30.29,-97.79,-97.71",
                "-o",
                str(tmp_path),
                "-lt",
                "Test",
            ],
        )

    assert result.exit_code == 0
    cfg = mock_run.call_args.args[0]
    assert cfg.zoom_bbox == [30.21, 30.29, -97.79, -97.71]


def test_run_cli_rejects_non_numeric_zoom_bbox(tmp_path):
    runner = CliRunner()

    with patch("disasters.cli.run_pipeline") as mock_run:
        result = runner.invoke(
            cli,
            [
                "run",
                "-b",
                "30.2 30.3 -97.8 -97.7",
                "-zb",
                "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                "-o",
                str(tmp_path),
                "-lt",
                "Test",
            ],
        )

    assert result.exit_code != 0
    assert "Failed to parse zoom bounding box" in result.output
    mock_run.assert_not_called()


def test_apply_slope_mask_to_uint8_without_source_nodata(tmp_path):
    """Masked uint8 rasters without source nodata should use a valid output nodata value."""
    target_tif = tmp_path / "target.tif"
    slope_tif = tmp_path / "slope.tif"
    output_tif = tmp_path / "output.tif"

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
    }

    with rasterio.open(target_tif, "w", **profile) as dst:
        dst.write(np.array([[1, 2], [3, 4]], dtype=np.uint8), 1)

    slope_profile = profile | {"dtype": "float32", "nodata": -9999}
    with rasterio.open(slope_tif, "w", **slope_profile) as dst:
        dst.write(np.array([[1, 20], [30, 2]], dtype=np.float32), 1)

    assert apply_slope_mask_to_raster(target_tif, slope_tif, 10, output_tif)

    with rasterio.open(output_tif) as src:
        data = src.read(1)
        assert src.nodata == 255

    assert data.tolist() == [[255, 2], [3, 255]]


def test_parse_bbox_input_preserves_url_aoi():
    url = "https://example.com/aoi.geojson"

    assert parse_bbox_input(url) == url


@patch("disasters.filters.gdal.DEMProcessing")
@patch("disasters.filters.gdal.Warp")
@patch("disasters.catalog.fetch_missing_dems")
@patch("disasters.pipeline.get_local_spatial_properties")
def test_process_dem_and_slope_fetches_missing_local_dems(
    mock_spatial, mock_fetch, mock_warp, mock_dem_processing, tmp_path
):
    local_wtr = tmp_path / "OPERA_L3_DSWX-HLS_T11ABC_20240101T000000Z_B01_WTR.tif"
    fetched_dem = tmp_path / "OPERA_L3_DSWX-HLS_T11ABC_20240101T000000Z_B10_DEM.tif"
    local_wtr.touch()

    df = pd.DataFrame(
        {
            "Dataset": ["OPERA_L3_DSWX-HLS_V1"],
            "Download URL WTR": [str(local_wtr)],
        }
    )
    mock_spatial.return_value = ([0, 1, 0, 1], "EPSG:4326")

    def create_dem(_bbox, output_dir):
        (output_dir / fetched_dem.name).touch()

    mock_fetch.side_effect = create_dem

    class FakeDataset:
        def ReadAsArray(self):
            return np.array([[5, 15]], dtype=np.float32)

    mock_warp.return_value = object()
    mock_dem_processing.return_value = FakeDataset()
    master_grid = {
        "shape": (1, 2),
        "transform": from_origin(0, 1, 1, 1),
        "dst_crs": "EPSG:4326",
    }

    mask = process_dem_and_slope(df, master_grid, 10, tmp_path)

    mock_fetch.assert_called_once()
    mock_warp.assert_called_once()
    assert str(fetched_dem) in mock_warp.call_args.args[1]
    assert mask.tolist() == [[True, False]]


@patch("disasters.pipeline.authenticate")
@patch("disasters.filters.apply_slope_mask_to_raster")
@patch("disasters.filters.process_dem_and_slope")
@patch("disasters.pipeline.get_master_grid_props")
@patch("disasters.pipeline.get_local_spatial_properties")
@patch("disasters.pipeline.scan_local_directory")
@patch("earthaccess.auth.Auth")
def test_run_slope_filter_only_processes_nested_tifs(
    mock_earthaccess_auth,
    mock_scan,
    mock_spatial,
    mock_grid,
    mock_process_slope,
    mock_apply_mask,
    mock_auth,
    tmp_path,
):
    nested_dir = tmp_path / "input" / "data"
    nested_dir.mkdir(parents=True)
    nested_tif = nested_dir / "OPERA_L2_RTC-S1_T11ABC_20240101T000000Z_VV.tif"
    nested_tif.touch()
    output_dir = tmp_path / "output"

    mock_earthaccess_auth.return_value.authenticated = True
    mock_scan.return_value = pd.DataFrame()
    mock_spatial.return_value = ([0, 1, 0, 1], "EPSG:4326")
    mock_grid.return_value = {
        "shape": (1, 1),
        "transform": from_origin(0, 1, 1, 1),
        "dst_crs": "EPSG:4326",
    }

    def create_slope(*_args, **kwargs):
        (kwargs["output_dir"] / "slope.tif").touch()
        return np.array([[False]])

    mock_process_slope.side_effect = create_slope
    mock_apply_mask.return_value = True

    assert run_slope_filter_only(tmp_path / "input", 10, output_dir) == output_dir

    mock_apply_mask.assert_called_once()
    assert mock_apply_mask.call_args.args[0] == nested_tif
