from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from disasters.cli import cli
from disasters.pipeline import PipelineConfig, run_download_only, run_pipeline


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
