import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.mask import mask

logger = logging.getLogger(__name__)

def compute_structure_impact(raster_path: Path, bbox_snwe: list, output_csv: Path) -> Tuple[int, int]:
    """
    Fetches OSM buildings for the bounding box, intersects them with the max flood extent raster
    using native rasterio masking, and saves the flooded structures to a CSV.

    Args:
        raster_path (Path): Path to the cumulative max_extent.tif (1=Water, 0=Dry, 255=NoData).
        bbox_snwe (list): Bounding box in [South, North, West, East] format.
        output_csv (Path): Path to save the resulting CSV of impacted buildings.

    Returns:
        tuple: (total_buildings_found, flooded_buildings_count)
    """
    logger.info("[Impact] Querying OpenStreetMap API for buildings in AOI...")
    
    s, n, w, e = bbox_snwe
    tags = {'building': True}
    
    try:
        # Fetch features from OSM
        gdf = ox.features_from_bbox(bbox=(n, s, e, w), tags=tags)
    except Exception as err:
        logger.warning(f"[Impact] OSM API query failed or timed out: {err}")
        return 0, 0

    if gdf.empty:
        logger.info("[Impact] No buildings found in this AOI via OSM.")
        return 0, 0

    # Filter to only keep polygons/multipolygons (buildings), drop nodes/lines
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    total_buildings = len(gdf)
    logger.info(f"[Impact] Retrieved {total_buildings} building polygons. Running spatial intersection...")

    # Read the CRS of the generated raster to align our vector data
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        
        # Project the OSM data (EPSG:4326) to the raster's UTM CRS
        gdf_proj = gdf.to_crs(raster_crs)
        
        # Define a quick helper function to check if a polygon touches water
        def is_flooded(geom):
            try:
                # Crop the raster to the exact shape of the building polygon
                out_image, _ = mask(src, [geom], crop=True)
                # Return True if any pixel inside the cropped array equals 1 (Water)
                return 1 if np.any(out_image == 1) else 0
            except ValueError:
                # Triggered if polygon falls completely outside raster bounds
                return 0

        # Apply the intersection logic row by row
        gdf_proj['flood_max'] = gdf_proj.geometry.apply(is_flooded)

    # Filter to only buildings that touched water
    flooded_gdf = gdf_proj[gdf_proj['flood_max'] == 1].copy()
    flooded_count = len(flooded_gdf)

    if flooded_count > 0:
        # Convert back to EPSG:4326 for human-readable Lat/Lon in the CSV
        flooded_gdf = flooded_gdf.to_crs("EPSG:4326")
        
        # Extract centroids for easy plotting/viewing
        flooded_gdf['longitude'] = flooded_gdf.geometry.centroid.x
        flooded_gdf['latitude'] = flooded_gdf.geometry.centroid.y
        
        # Clean up the dataframe before saving
        cols_to_keep = ['longitude', 'latitude']
        if 'name' in flooded_gdf.columns: cols_to_keep.append('name')
        if 'building' in flooded_gdf.columns: cols_to_keep.append('building')
        
        df_out = flooded_gdf[cols_to_keep]
        df_out.to_csv(output_csv, index=False)
        logger.info(f"[Impact] Saved {flooded_count} impacted structures to {output_csv.name}")
    else:
        logger.info("[Impact] Zero buildings intersected the flood extent.")

    return total_buildings, flooded_count