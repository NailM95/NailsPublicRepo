"""
3D building-height map for Baku.

Data: OpenStreetMap building footprints (height / building:levels tags) via OSMnx,
      which queries the public Overpass API. Basemap tiles come from CARTO.
Each building is extruded to its height and colored blue (low) -> red (tall).
Hovering a building shows its name, address and height (from OSM address tags).

Install:
    pip install streamlit "osmnx>=2.0" geopandas pydeck matplotlib
Run:
    streamlit run baku_tall_buildings.py
"""
import html
import json

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pydeck as pdk
import streamlit as st
from matplotlib import colormaps

st.set_page_config(page_title="Baku building heights", layout="wide")

# West, South, East, North — central Baku. Widen it for the whole city.
DEFAULT_BBOX = (49.78, 40.35, 49.97, 40.44)
UTM_BAKU = 32639  # UTM zone 39N, metric CRS for area calculations
UNKNOWN_GREY = [190, 190, 190]


def to_number(series: pd.Series) -> pd.Series:
    """Pull the first number out of OSM tag strings like '45 m', '12;14', '3,5'."""
    extracted = series.astype(str).str.extract(r"(\d+(?:[.,]\d+)?)")[0]
    return pd.to_numeric(extracted.str.replace(",", "."), errors="coerce")


def build_address(df: pd.DataFrame) -> pd.Series:
    """'Street HouseNumber', falling back to addr:full; empty string if nothing is tagged."""
    for col in ("addr:street", "addr:housenumber", "addr:full"):
        if col not in df.columns:
            df[col] = np.nan
    street = df["addr:street"].fillna("").astype(str).str.strip()
    number = df["addr:housenumber"].fillna("").astype(str).str.strip()
    addr = (street + " " + number).str.strip()
    return addr.where(addr != "", df["addr:full"].fillna("").astype(str))


def fill_addresses_from_points(buildings: gpd.GeoDataFrame, bbox: tuple) -> gpd.GeoDataFrame:
    """In OSM the address is often a separate point inside the building, not a tag on it.
    Give buildings without an address the address of a point that sits inside them."""
    try:
        points = ox.features_from_bbox(bbox=bbox, tags={"addr:housenumber": True})
    except Exception:  # no address points in the box, or Overpass hiccup
        return buildings
    points = points[points.geometry.geom_type == "Point"].copy()
    if points.empty:
        return buildings
    points["addr_pt"] = build_address(points)
    points = points[points["addr_pt"] != ""][["geometry", "addr_pt"]].reset_index(drop=True)

    missing = buildings[buildings["address"] == ""]
    joined = gpd.sjoin(points, missing[["geometry"]], predicate="within")
    first = joined.groupby("index_right")["addr_pt"].first()
    buildings.loc[first.index, "address"] = first.values
    return buildings


@st.cache_data(show_spinner="Downloading buildings from OpenStreetMap…")
def load_buildings(bbox: tuple, floor_height: float) -> gpd.GeoDataFrame:
    gdf = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    for col in ("height", "building:levels"):
        if col not in gdf.columns:
            gdf[col] = np.nan

    height = to_number(gdf["height"])
    levels = to_number(gdf["building:levels"])
    gdf["height_m"] = height.fillna(levels * floor_height)
    gdf["height_source"] = np.select(
        [height.notna(), levels.notna()], ["height tag", "levels tag"], "unknown"
    )
    gdf["area_m2"] = gdf.to_crs(UTM_BAKU).area.values
    gdf["name"] = gdf["name"].fillna("").astype(str) if "name" in gdf.columns else ""
    gdf["address"] = build_address(gdf)

    cols = ["geometry", "height_m", "height_source", "area_m2", "name", "address"]
    gdf = gpd.GeoDataFrame(gdf[cols].reset_index(drop=True), geometry="geometry", crs=4326)
    return fill_addresses_from_points(gdf, bbox)


def colorize(values, clip_pct: float):
    """Map values to RGB; RdYlBu_r runs blue (low) -> yellow -> red (high)."""
    v = pd.Series(values).fillna(0).to_numpy(dtype=float)
    hi = np.percentile(v, clip_pct) if v.size else 0
    norm = np.clip(v / hi, 0, 1) if hi > 0 else np.zeros_like(v)
    return (colormaps["RdYlBu_r"](norm)[:, :3] * 255).astype(int).tolist()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    bbox_text = st.text_input("Bounding box (W, S, E, N)", ",".join(map(str, DEFAULT_BBOX)))
    bbox = tuple(float(x) for x in bbox_text.split(","))
    floor_height = st.slider("Assumed floor height (m)", 2.5, 4.0, 3.0, 0.1)
    min_height = st.slider("Show buildings from (m)", 0, 150, 0, 5,
                           help="Hide lower buildings to make the towers stand out.")
    show_unknown = st.checkbox("Show buildings without height data (grey)", value=True)
    clip_pct = st.slider("Color scale cap (percentile)", 80, 100, 98,
                         help="Stops a few very tall towers from turning everything else blue.")
    opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05)

# ---------------------------------------------------------------- data + coverage
buildings = load_buildings(bbox, floor_height)

known = (buildings.height_source != "unknown").mean() if len(buildings) else 0
c1, c2, c3 = st.columns(3)
c1.metric("Buildings loaded", f"{len(buildings):,}")
c2.metric("With height or levels", f"{known:.0%}")
c3.metric("With address", f"{(buildings.address != '').mean():.0%}")
if known < 0.5:
    st.warning("Less than half of buildings have height data in OpenStreetMap, "
               "so grey buildings may include tall ones that nobody has tagged yet.")

# ---------------------------------------------------------------- layer
has_height = buildings.height_m.notna()
keep = (has_height & (buildings.height_m >= min_height)) | (~has_height & show_unknown)
shown = buildings.loc[keep, ["geometry", "height_m", "height_source", "name", "address"]].copy()

colors = colorize(shown["height_m"], clip_pct)
shown["color"] = [c if not np.isnan(h) else UNKNOWN_GREY
                  for c, h in zip(colors, shown["height_m"])]
shown["elev"] = shown["height_m"].fillna(3.0)


def tooltip_html(name: str, address: str, height: float, source: str) -> str:
    lines = []
    if name:
        lines.append(f"<b>{html.escape(name)}</b>")
    lines.append(html.escape(address) if address else "<i>No address in OpenStreetMap</i>")
    lines.append("Height: unknown" if np.isnan(height)
                 else f"Height: {height:.0f} m <span style='opacity:.7'>({source})</span>")
    return "<br>".join(lines)


shown["info"] = [tooltip_html(n, a, h, src) for n, a, h, src in
                 zip(shown["name"], shown["address"], shown["height_m"], shown["height_source"])]
shown = shown[["geometry", "color", "elev", "info"]]

layer = pdk.Layer(
    "GeoJsonLayer",
    json.loads(shown.to_json()),
    get_fill_color="properties.color",
    extruded=True,
    get_elevation="properties.elev",
    pickable=True,
    opacity=opacity,
    wireframe=False,
)
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=(bbox[1] + bbox[3]) / 2, longitude=(bbox[0] + bbox[2]) / 2,
        zoom=13.5, pitch=50, bearing=-20,
    ),
    map_provider="carto",
    map_style="light",
    tooltip={"html": "{info}",
             "style": {"fontSize": "13px", "maxWidth": "280px", "lineHeight": "1.4"}},
)
st.pydeck_chart(deck, height=720)

st.download_button("Download map as HTML", deck.to_html(as_string=True),
                   file_name="baku_building_heights.html", mime="text/html")