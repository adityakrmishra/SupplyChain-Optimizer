import folium
from folium.plugins import (MarkerCluster, HeatMap, MeasureControl, 
                          TimestampedGeoJson, Fullscreen, FloatImage)
import geopandas as gpd
from typing import List, Tuple, Optional, Dict, Union
from datetime import datetime, timedelta
import requests
import json

class AdvancedSupplyChainMap:
    """Advanced interactive mapping for supply chain optimization with real-time features"""
    
    def __init__(self, location: Tuple[float, float] = (20.5937, 78.9629), 
                 zoom_start: int = 5, tiles: str = 'cartodbpositron'):
        """
        Initialize advanced supply chain visualization map
        
        Args:
            location: Initial map center coordinates
            zoom_start: Initial zoom level
            tiles: Base tile layer
        """
        self.base_map = folium.Map(location=location, zoom_start=zoom_start, 
                                 tiles=tiles, control_scale=True)
        
        # Configure multiple tile layers
        self._add_tile_layers()
        
        # Initialize feature groups
        self.feature_groups = {
            'routes': folium.FeatureGroup(name='Optimized Routes', show=True),
            'proposed_routes': folium.FeatureGroup(name='Proposed Routes', show=False),
            'warehouses': folium.FeatureGroup(name='Warehouse Network'),
            'disruptions': folium.FeatureGroup(name='Risk Zones'),
            'sensors': MarkerCluster(name='IoT Sensors'),
            'suppliers': folium.FeatureGroup(name='Supplier Locations')
        }
        
        for group in self.feature_groups.values():
            group.add_to(self.base_map)
            
        # Add map controls
        self._add_map_controls()
        
        # Initialize real-time data structures
        self.realtime_data = {
            'weather': [],
            'shipments': []
        }

    def _add_tile_layers(self) -> None:
        """Add alternative base map layers"""
        tile_layers = {
            'Satellite': folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite Imagery',
                overlay=False
            ),
            'Terrain': folium.TileLayer(
                tiles='Stamen Terrain',
                attr='Stamen',
                name='Terrain Map'
            )
        }
        
        for layer in tile_layers.values():
            layer.add_to(self.base_map)

    def _add_map_controls(self) -> None:
        """Add interactive map controls"""
        # Measurement tools
        MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(self.base_map)
        
        # Fullscreen control
        Fullscreen(position='topright').add_to(self.base_map)
        
        # Layer control
        folium.LayerControl(collapsed=False, position='bottomright').add_to(self.base_map)
        
        # Add logo
        FloatImage('https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon.png',
                 position='bottomleft', width='50px').add_to(self.base_map)

    def add_optimized_route(self, coordinates: List[Tuple[float, float]], 
                          route_data: Dict[str, Union[float, str]]) -> None:
        """
        Add optimized route with performance metrics
        
        Args:
            coordinates: List of (lat, lon) tuples
            route_data: Dictionary containing route metadata
                       (distance, eta, cost, carbon_footprint)
        """
        # Create polyline
        route_line = folium.PolyLine(
            locations=coordinates,
            color='#00cc00' if route_data.get('optimized', False) else '#ff3300',
            weight=6,
            opacity=0.8,
            dash_array='5' if not route_data.get('optimized') else None
        )
        
        # Create interactive popup
        popup_html = f"""
        <div style="width: 250px;">
            <h4>Route Performance</h4>
            <table>
                <tr><td>Distance:</td><td>{route_data.get('distance_km', 0)} km</td></tr>
                <tr><td>ETA:</td><td>{route_data.get('eta_hours', 0)} hrs</td></tr>
                <tr><td>Cost:</td><td>${route_data.get('cost', 0):.2f}</td></tr>
                <tr><td>CO₂:</td><td>{route_data.get('carbon_footprint', 0)} kg</td></tr>
            </table>
        </div>
        """
        
        route_line.add_child(folium.Popup(popup_html))
        route_line.add_to(self.feature_groups['routes'])

    def add_warehouse_complex(self, location: Tuple[float, float], 
                            warehouse_info: Dict[str, Union[float, str]]) -> None:
        """
        Add advanced warehouse visualization with capacity metrics
        
        Args:
            location: (lat, lon) coordinates
            warehouse_info: Dictionary containing warehouse metadata
        """
        # Custom icon based on warehouse type
        icon_config = {
            'cold_storage': {'icon': 'snowflake', 'color': 'blue', 'prefix': 'fa'},
            'hazardous': {'icon': 'radiation', 'color': 'orange', 'prefix': 'fa'},
            'fulfillment': {'icon': 'box-open', 'color': 'green', 'prefix': 'fa'},
            'default': {'icon': 'warehouse', 'color': 'gray', 'prefix': 'fa'}
        }
        
        w_type = warehouse_info.get('type', 'default')
        icon = icon_config.get(w_type, icon_config['default'])
        
        # Create marker
        warehouse_marker = folium.Marker(
            location=location,
            icon=folium.Icon(**icon),
            tooltip=f"Warehouse: {warehouse_info.get('name', '')}"
        )
        
        # Create detailed popup
        popup_html = f"""
        <div style="width: 300px;">
            <h3>{warehouse_info.get('name', 'Unnamed Warehouse')}</h3>
            <p><b>Type:</b> {w_type.replace('_', ' ').title()}</p>
            <p><b>Capacity:</b> {warehouse_info.get('capacity_sqft', 0)} sqft</p>
            <p><b>Utilization:</b> {warehouse_info.get('utilization', 0)}%</p>
            <p><b>Last Audit:</b> {warehouse_info.get('last_audit', 'N/A')}</p>
            <hr>
            <p><i>{warehouse_info.get('status', 'Operational')}</i></p>
        </div>
        """
        
        warehouse_marker.add_child(folium.Popup(popup_html))
        warehouse_marker.add_to(self.feature_groups['warehouses'])

    def add_risk_zone(self, geojson_data: Dict, 
                     risk_type: str = 'flood') -> None:
        """
        Add geospatial risk zone from GeoJSON data
        
        Args:
            geojson_data: GeoJSON feature collection
            risk_type: Type of risk (flood, earthquake, political)
        """
        style_map = {
            'flood': {'fillColor': '#0066cc', 'color': '#0066cc'},
            'earthquake': {'fillColor': '#ff0000', 'color': '#ff0000'},
            'political': {'fillColor': '#ff9900', 'color': '#ff9900'}
        }
        
        folium.GeoJson(
            geojson_data,
            name=f'{risk_type.title()} Risk Zone',
            style_function=lambda x: style_map.get(risk_type, {}),
            tooltip=folium.GeoJsonTooltip(fields=['name', 'risk_level'])
        ).add_to(self.feature_groups['disruptions'])

    def add_real_time_weather(self, api_key: str) -> None:
        """Integrate real-time weather data from OpenWeatherMap API"""
        # Example implementation - would need proper API integration
        weather_data = self._fetch_weather_data(api_key)
        
        for report in weather_data:
            self._add_weather_marker(report)

    def _fetch_weather_data(self, api_key: str) -> List[Dict]:
        """Simulate weather data fetching (replace with actual API call)"""
        return [{
            'lat': 19.0760,
            'lon': 72.8777,
            'conditions': 'thunderstorm',
            'intensity': 'severe'
        }]

    def _add_weather_marker(self, weather_report: Dict) -> None:
        """Add weather disruption marker to map"""
        icon = folium.Icon(
            icon='cloud-showers-heavy',
            prefix='fa',
            color='lightgray'
        )
        
        folium.Marker(
            location=(weather_report['lat'], weather_report['lon']),
            icon=icon,
            popup=f"Weather Alert: {weather_report['conditions'].title()}",
            tooltip="Click for weather details"
        ).add_to(self.feature_groups['disruptions'])

    def add_temporal_shipments(self, shipment_data: List[Dict]) -> None:
        """
        Add time-animated shipment visualization
        
        Args:
            shipment_data: List of shipment records with timestamps
        """
        features = []
        for shipment in shipment_data:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [shipment['lon'], shipment['lat']]
                },
                'properties': {
                    'time': shipment['timestamp'],
                    'style': {'color': shipment.get('color', '#ff0000')},
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': shipment.get('color', '#ff0000'),
                        'fillOpacity': 0.8,
                        'stroke': False,
                        'radius': 8
                    },
                    'popup': f"Shipment ID: {shipment['id']}<br>Status: {shipment['status']}"
                }
            }
            features.append(feature)
            
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='PT1H',  # Update interval
            add_last_point=True,
            auto_play=True,
            loop=False,
            max_speed=5,
            loop_button=True,
            date_options='YYYY/MM/DD HH:mm:ss',
            transition_time=500
        ).add_to(self.base_map)

    def generate_supply_chain_dashboard(self, output_file: str = 'supply_chain_dashboard.html') -> None:
        """Save complete map visualization with all elements"""
        self.base_map.save(output_file)

def visualize_geodataframe(gdf: gpd.GeoDataFrame, 
                          color: str = 'blue',
                          column: Optional[str] = None,
                          legend_name: str = '') -> folium.Map:
    """
    Create choropleth map from GeoDataFrame
    
    Args:
        gdf: GeoDataFrame with geometry column
        color: Base color or column name for choropleth
        column: Data column for color mapping
        legend_name: Legend title
    
    Returns:
        folium.Map: Configured map object
    """
    centroid = gdf.geometry.centroid.unary_union.centroid.coords[0]
    m = folium.Map(location=[centroid[1], centroid[0]], zoom_start=6)
    
    if column:
        # Choropleth visualization
        folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            data=gdf,
            columns=[gdf.index.name, column],
            key_on='feature.id',
            fill_color='YlGnBu',
            legend_name=legend_name,
            highlight=True
        ).add_to(m)
    else:
        # Simple GeoJSON layer
        folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6
            }
        ).add_to(m)
    
    return m

# Example Usage
if __name__ == "__main__":
    # Initialize advanced map
    sc_map = AdvancedSupplyChainMap(location=[28.6139, 77.2090], zoom_start=5)
    
    # Add optimized route
    route_coords = [
        (28.6139, 77.2090),  # Delhi
        (19.0760, 72.8777),   # Mumbai
        (13.0827, 80.2707)    # Chennai
    ]
    sc_map.add_optimized_route(route_coords, {
        'distance_km': 2200,
        'eta_hours': 48,
        'cost': 4500.00,
        'carbon_footprint': 1200,
        'optimized': True
    })
    
    # Add warehouse
    sc_map.add_warehouse_complex((19.0760, 72.8777), {
        'name': 'Mumbai Cold Storage',
        'type': 'cold_storage',
        'capacity_sqft': 150000,
        'utilization': 85,
        'last_audit': '2024-02-15',
        'status': 'High Capacity'
    })
    
    # Generate dashboard
    sc_map.generate_supply_chain_dashboard()
