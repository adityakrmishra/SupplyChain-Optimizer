import pytest
import geopandas as gpd
from shapely.geometry import Point
from utils.geo_helpers import GeoUtils

@pytest.fixture
def sample_gdf():
    return gpd.GeoDataFrame({
        'city': ['Berlin', 'Paris'],
        'lat': [52.5200, 48.8566],
        'lon': [13.4050, 2.3522]
    }, geometry=[Point(13.4050, 52.5200), Point(2.3522, 48.8566)])

def test_distance_matrix():
    coords = [(52.5200, 13.4050), (48.8566, 2.3522)]
    matrix = GeoUtils.calculate_distance_matrix(coords)
    
    assert matrix.shape == (2, 2)
    assert pytest.approx(matrix.iloc[0,1], 878)  # Berlin-Paris distance
    assert matrix.iloc[0,0] == 0

def test_geodf_conversion(sample_gdf):
    converted = GeoUtils.convert_to_geodf(sample_gdf)
    assert isinstance(converted, gpd.GeoDataFrame)
    assert converted.crs == "EPSG:4326"

def test_geojson_merging(tmp_path):
    file1 = tmp_path / "test1.geojson"
    file2 = tmp_path / "test2.geojson"
    
    gdf1 = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    gdf2 = gpd.GeoDataFrame(geometry=[Point(1, 1)])
    
    gdf1.to_file(file1, driver='GeoJSON')
    gdf2.to_file(file2, driver='GeoJSON')
    
    merged = GeoUtils.merge_geojsons([file1, file2])
    assert len(merged) == 2
