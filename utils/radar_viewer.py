#!/usr/bin/env python3
"""
Radar / GeoTIFF visualizer with weather-radar false-color LUT
and dBZ intervals defined in an external JSON palette file.

- Supports standard TIFF (via Pillow)
- Supports GeoTIFF (via rasterio, if installed)
- Allows band selection for multi-band GeoTIFFs
- Prints GeoTIFF metadata (CRS, resolution, bounds)
"""

import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Try to import rasterio for GeoTIFF support
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# -------------------------------------------------------------------------
# Load palette (colors + dBZ levels) from an external JSON file
# -------------------------------------------------------------------------
def load_palette_from_json(path: Path):
    """
    Load colors, dBZ levels, and an optional label from a JSON file.

    Expected JSON structure:
    {
        "levels": [-10, -5, 0, 5, ...],
        "colors": ["#646464", "#04e9e7", "..."],
        "label": "Reflectivity (dBZ)"   # optional
    }

    Returns:
        (cmap, levels_array, label)
    """
    if not path.exists():
        raise FileNotFoundError(f"Palette JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        palette_data = json.load(f)

    levels = palette_data.get("levels")
    colors = palette_data.get("colors")

    if not isinstance(levels, list) or len(levels) == 0:
        raise ValueError(
            f'Palette JSON "{path}" must contain a non-empty "levels" list.'
        )

    if not isinstance(colors, list) or len(colors) == 0:
        raise ValueError(
            f'Palette JSON "{path}" must contain a non-empty "colors" list.'
        )

    levels_array = np.array(levels, dtype=float)
    label = palette_data.get("label", "Reflectivity (dBZ)")

    if not (len(colors) == len(levels) or len(colors) == len(levels) - 1):
        print(
            f"Warning: palette JSON '{path}' has {len(colors)} colors and "
            f"{len(levels)} levels. Check if this is intended."
        )

    cmap = ListedColormap(colors)
    return cmap, levels_array, label


# -------------------------------------------------------------------------
# Load TIFF / GeoTIFF and return data + optional extent
# -------------------------------------------------------------------------
def load_raster(path: Path, band: int):
    """
    Load a raster file.

    If rasterio is available and the file is a GeoTIFF, it will:
      - load the selected band (1-based index)
      - return a (data, extent) tuple, where extent = (xmin, xmax, ymin, ymax)
      - print GeoTIFF metadata (CRS, resolution, bounds)

    Otherwise, it falls back to Pillow and returns:
      - data as a 2D NumPy array
      - extent = None
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try rasterio path first if available (GeoTIFF)
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(path) as src:
                # Print metadata
                print(f"GeoTIFF detected: {path.name}")
                print(f"  CRS: {src.crs}")
                print(f"  Resolution: {src.res}")
                print(
                    f"  Bounds: left={src.bounds.left}, right={src.bounds.right}, "
                    f"bottom={src.bounds.bottom}, top={src.bounds.top}"
                )

                # Read selected band (1-based index)
                data = src.read(band)

                # Define extent in map units (for imshow)
                bounds = src.bounds
                extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
                return data, extent
        except Exception as e:
            print(f"Warning: rasterio failed to open '{path}' as GeoTIFF or read band {band}: {e}")
            print("Falling back to Pillow...")

    # Fallback: use Pillow (no georeferencing)
    im = Image.open(path)
    data = np.array(im)
    extent = None
    return data, extent


# -------------------------------------------------------------------------
# Render and optionally save the radar / geotiff plot
# -------------------------------------------------------------------------
def plot_radar(
    data: np.ndarray,
    cmap: ListedColormap,
    levels: np.ndarray,
    title: str,
    show_colorbar: bool,
    colorbar_label: str,
    save_path: Path | None,
    dpi: int,
    extent=None,
    show_axes: bool = False,
):
    """
    Plot the raster using the given colormap and dBZ levels.
    """
    norm = BoundaryNorm(levels, cmap.N)

    plt.figure(figsize=(6, 6), dpi=dpi)

    if extent is not None:
        img = plt.imshow(data, cmap=cmap, norm=norm, origin="upper", extent=extent)
    else:
        img = plt.imshow(data, cmap=cmap, norm=norm, origin="upper")

    if not show_axes:
        plt.axis("off")

    plt.title(title)

    if show_colorbar:
        cbar = plt.colorbar(img, ticks=levels)
        cbar.set_label(colorbar_label)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    plt.close()


# -------------------------------------------------------------------------
# Command-line interface argument parsing
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a radar TIFF / GeoTIFF using a weather-radar false-color LUT "
            "and dBZ intervals defined in an external JSON palette file."
        )
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input TIFF / GeoTIFF file."
    )

    parser.add_argument(
        "-p", "--palette",
        type=Path,
        required=True,
        help="Path to the JSON file defining colors and dBZ levels."
    )

    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Band index to read (1-based, GeoTIFF only). Default: 1."
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Weather Radar-style False Color Image",
        help="Title of the plot."
    )

    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the colorbar."
    )

    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save output figure to file instead of showing."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Image resolution (default: 100 dpi)."
    )

    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Keep axes visible (useful for GeoTIFF coordinates)."
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Main runtime entry point
# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Load raster (GeoTIFF via rasterio or plain TIFF via Pillow)
    data, extent = load_raster(args.input, band=args.band)

    # Load colormap, dBZ levels, and label from JSON palette
    cmap, levels, colorbar_label = load_palette_from_json(args.palette)

    # Plot using palette + levels + optional GeoTIFF extent
    plot_radar(
        data=data,
        cmap=cmap,
        levels=levels,
        title=args.title,
        show_colorbar=not args.no_colorbar,
        colorbar_label=colorbar_label,
        save_path=args.save,
        dpi=args.dpi,
        extent=extent,
        show_axes=args.show_axes,
    )


if __name__ == "__main__":
    main()

