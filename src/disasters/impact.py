import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from osmnx._errors import InsufficientResponseError
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.windows import Window
from shapely.geometry import box, shape

logger = logging.getLogger(__name__)

# Overpass request timeout in seconds (osmnx >= 2.0 name; "timeout" was removed)
ox.settings.requests_timeout = 900


def _write_empty_impact_outputs(output_csv: Path) -> None:
    """
    Write header-only CSV and empty GeoJSON outputs, so a completed analysis with no
    impacted structures is distinguishable from one that never ran.

    Args:
        output_csv (Path): Path of the impacted-structures CSV (GeoJSON uses the same stem).
    """
    pd.DataFrame(columns=["longitude", "latitude"]).to_csv(output_csv, index=False)
    output_csv.with_suffix(".geojson").write_text(
        '{"type": "FeatureCollection", "features": []}\n'
    )


def compute_structure_impact(
    raster_path: Path, bbox_snwe: list, output_csv: Path
) -> Tuple[int, int]:
    """
    Fetches OSM buildings for the bounding box, intersects them with the max flood extent raster
    using native rasterio masking, and saves the flooded structures to a CSV.

    Args:
        raster_path (Path): Path to the cumulative max_extent.tif (1=Water, 0=Dry, 255=NoData).
        bbox_snwe (list): Bounding box in [South, North, West, East] format.
        output_csv (Path): Path to save the resulting CSV of impacted buildings. The CSV and a
            matching GeoJSON are always written (empty when no structures are impacted).

    Returns:
        tuple: (total_buildings_found, flooded_buildings_count). total_buildings_found counts
            the OSM buildings retrieved within raster windows that contain flood pixels.
            If any OSM requests fail, both counts are lower bounds and a warning is logged.
    """
    logger.info("[Impact] Analyzing max flood extent using smart grid chunking...")

    gdf_list = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        transform = src.transform
        transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

        # Define window size (500x500 pixels = 15x15km at 30m resolution)
        window_size = 500

        height = src.height
        width = src.width

        total_windows = int(
            np.ceil(height / window_size) * np.ceil(width / window_size)
        )
        wet_windows = 0
        failed_windows = 0
        current_window = 0

        # Iterate through the raster in 15km x 15km blocks
        for row_off in range(0, height, window_size):
            for col_off in range(0, width, window_size):
                current_window += 1

                # Protect against edge-of-raster bounds
                w_width = min(window_size, width - col_off)
                w_height = min(window_size, height - row_off)

                window = Window(col_off, row_off, w_width, w_height)
                chunk = src.read(1, window=window)

                # If there's no water (1) in this chunk, skip the API call entirely!
                if not np.any(chunk == 1):
                    continue

                wet_windows += 1

                # Print progress to the console
                logger.info(
                    f"[Impact] Fetching OSM data for flooded chunk {wet_windows} (Grid window {current_window}/{total_windows})..."
                )

                # Get transform
                left = transform.c + col_off * transform.a
                right = transform.c + (col_off + w_width) * transform.a
                top = transform.f + row_off * transform.e
                bottom = transform.f + (row_off + w_height) * transform.e

                # Project to Lat/Lon
                lon_left, lat_bottom = transformer.transform(left, bottom)
                lon_right, lat_top = transformer.transform(right, top)

                # Ensure absolute min/max ordering regardless of hemisphere
                min_lon, max_lon = min(lon_left, lon_right), max(lon_left, lon_right)
                min_lat, max_lat = min(lat_bottom, lat_top), max(lat_bottom, lat_top)

                # Create a Shapely polygon
                poly = box(min_lon, min_lat, max_lon, max_lat)

                # Fetch from OSM using the polygon
                try:
                    chunk_gdf = ox.features_from_polygon(poly, tags={"building": True})
                    if not chunk_gdf.empty:
                        # Keep only actual building footprints
                        chunk_gdf = chunk_gdf[
                            chunk_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
                        ]
                        if not chunk_gdf.empty:
                            gdf_list.append(chunk_gdf)
                except InsufficientResponseError:
                    # OSM returned no buildings for this window (not a failure)
                    continue
                except Exception as e:
                    failed_windows += 1
                    logger.warning(
                        f"[Impact] OSM request failed for window {current_window}/{total_windows}: {e}"
                    )
                    continue

    logger.info(
        f"[Impact] Skipped {total_windows - wet_windows} non-flooded windows. Queried OSM for {wet_windows} flooded windows."
    )
    if failed_windows:
        logger.warning(
            f"[Impact] {failed_windows}/{wet_windows} flooded windows failed to download OSM buildings. "
            "Impacted structure counts are INCOMPLETE (lower bound)."
        )

    if not gdf_list:
        logger.info("[Impact] No buildings found in the flooded areas.")
        _write_empty_impact_outputs(output_csv)
        return 0, 0

    # Combine all chunked building dataframes
    gdf = pd.concat(gdf_list)

    # Drop duplicates (in case a building footprint crossed the boundary of two windows)
    gdf = gdf[~gdf.index.duplicated(keep="first")].copy()

    logger.info(
        f"[Impact] Retrieved {len(gdf)} buildings near water. Running spatial intersection..."
    )

    # Project the OSM data (EPSG:4326) to the raster's UTM CRS for accurate intersection
    gdf_proj = gdf.to_crs(raster_crs)

    with rasterio.open(raster_path) as src:

        def is_flooded(geom):
            try:
                # Use all_touched=True to ensure we grab ANY water pixel touching the building
                out_image, out_transform = mask(
                    src, [geom], crop=True, all_touched=True
                )
                flood_data = out_image[0]

                # If no water (1) touches the building at all, skip
                if not np.any(flood_data == 1):
                    return 0

                # Isolate water pixels to convert to geometry
                water_mask = (flood_data == 1).astype("uint8")

                # Convert the raster water pixels into exact geometric squares (polygons)
                water_polys = []
                for geom_dict, val in shapes(water_mask, transform=out_transform):
                    if val == 1:
                        water_polys.append(shape(geom_dict))

                if not water_polys:
                    return 0

                # Merge the water squares into a single unified geometry
                total_water_geom = gpd.GeoSeries(water_polys).union_all()

                # Calculate the true fractional area intersection!
                intersection_area = geom.intersection(total_water_geom).area
                building_area = geom.area

                coverage = intersection_area / building_area

                # Consider the building flooded if at least 50% of its TRUE area is covered
                return 1 if coverage >= 0.5 else 0
            except Exception:
                return 0

        # Apply the true geometric intersection logic row by row
        gdf_proj["flood_max"] = gdf_proj.geometry.apply(is_flooded)

    # Filter to only buildings that touched water
    flooded_gdf = gdf_proj[gdf_proj["flood_max"] == 1].copy()
    flooded_count = len(flooded_gdf)
    total_buildings = len(gdf_proj)

    if flooded_count > 0:
        # Calculate centroids in UTM meters
        centroids_utm = flooded_gdf.geometry.centroid

        # Convert both the building polygons and the centroids back to Lat/Lon
        flooded_gdf = flooded_gdf.to_crs("EPSG:4326")
        centroids_4326 = centroids_utm.to_crs("EPSG:4326")

        # Save geojson of building polygons
        geojson_path = output_csv.with_suffix(".geojson")
        geo_cols = ["geometry"]
        if "name" in flooded_gdf.columns:
            geo_cols.append("name")
        if "building" in flooded_gdf.columns:
            geo_cols.append("building")

        # Save spatial file
        flooded_gdf[geo_cols].to_file(geojson_path, driver="GeoJSON")
        logger.info(
            f"[Impact] Saved {flooded_count} impacted structure footprints to {geojson_path.name}"
        )

        # Save csv of building centroids
        flooded_gdf["longitude"] = centroids_4326.x
        flooded_gdf["latitude"] = centroids_4326.y

        csv_cols = ["longitude", "latitude"]
        if "name" in flooded_gdf.columns:
            csv_cols.append("name")
        if "building" in flooded_gdf.columns:
            csv_cols.append("building")

        # Save flat file
        df_out = flooded_gdf[csv_cols]
        df_out.to_csv(output_csv, index=False)
        logger.info(
            f"[Impact] Saved {flooded_count} impacted structure centroids to {output_csv.name}"
        )

    else:
        logger.info("[Impact] Zero buildings intersected the flood extent.")
        _write_empty_impact_outputs(output_csv)

    return total_buildings, flooded_count
