from geopy.distance import great_circle
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import folium
from typing import List, Tuple

class GeoUtils:
    """Geospatial helper functions for supply chain analysis"""
    
    @staticmethod
    def calculate_distance_matrix(coords: List[Tuple[float, float]]) -> pd.DataFrame:
        """Create distance matrix from coordinates"""
        return pd.DataFrame(
            [[great_circle(i, j).km for j in coords] for i in coords],
            index=coords,
            columns=coords
        )
    
    @staticmethod
    def convert_to_geodf(df: pd.DataFrame, 
                        lat_col: str = 'lat', 
                        lon_col: str = 'lon') -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame"""
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    @staticmethod
    def merge_geojsons(files: List[str]) -> gpd.GeoDataFrame:
        """Combine multiple GeoJSON files"""
        return gpd.GeoDataFrame(
            pd.concat([gpd.read_file(f) for f in files], 
                      ignore_index=True)
        )
    
    @staticmethod
    def plot_geodata(gdf: gpd.GeoDataFrame, 
                    color_column: str = None) -> folium.Map:
        """Create interactive Folium map from GeoDataFrame"""
        centroid = gdf.geometry.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=8)
        
        for _, row in gdf.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    'color': 'blue' if not color_column else x['properties'][color_column]
                }
            ).add_to(m)
        
        return m
