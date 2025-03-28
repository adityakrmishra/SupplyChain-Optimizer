import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon
from typing import Optional, Tuple, List, Dict, Union
from loguru import logger
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor

ox.config(use_cache=True, log_console=True)

class AdvancedLogisticsNetwork:
    """Enhanced logistics network analyzer with multi-modal support"""
    
    def __init__(self, network_type: str = 'drive_service', 
                 custom_filters: Optional[str] = None):
        """
        Initialize logistics network analyzer
        
        Args:
            network_type: OSMnx network type
            custom_filters: Custom OSM filter (e.g., '["highway"~"motorway|trunk"]')
        """
        self.network_type = network_type
        self.custom_filters = custom_filters
        self.graph = None
        self.edge_speeds = None
        self._vehicle_profiles = {
            'heavy_truck': {
                'max_weight': 40,  # tons
                'height': 4.5,     # meters
                'hazardous': True,
                'speed_factor': 0.8
            },
            'van': {
                'max_weight': 3,
                'height': 2.5,
                'speed_factor': 1.0
            }
        }

    def load_network(self, location: Union[str, Tuple[float, float, float, float]], 
                    dist: int = 10000) -> None:
        """
        Load transportation network with advanced configuration
        
        Args:
            location: Address, coordinates, or bounding box
            dist: Distance in meters if using point coordinates
        """
        try:
            if isinstance(location, tuple):
                self.graph = ox.graph_from_point(
                    (location[0], location[1]),
                    dist=dist,
                    network_type=self.network_type,
                    custom_filter=self.custom_filters,
                    infrastructure='way["highway"]'
                )
            else:
                self.graph = ox.graph_from_place(
                    location,
                    network_type=self.network_type,
                    custom_filter=self.custom_filters
                )
                
            self._preprocess_network()
            logger.success(f"Loaded network with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Network loading failed: {str(e)}")
            raise

    def _preprocess_network(self) -> None:
        """Apply network preprocessing steps"""
        # Add edge speed and travel time attributes
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)
        
        # Add elevation data if available
        if 'elevation' not in self.graph.nodes[0]:
            try:
                self.graph = ox.elevation.add_node_elevations_google(
                    self.graph, 
                    api_key=os.getenv('GOOGLE_ELEVATION_API_KEY')
                )
            except Exception:
                logger.warning("Elevation data unavailable")
                
        # Calculate edge grades
        self.graph = ox.elevation.add_edge_grades(self.graph)

    def optimize_multiobjective_route(self, 
                                     origin: Tuple[float, float],
                                     destination: Tuple[float, float],
                                     objectives: List[Dict[str, float]] = [
                                         {'weight': 'travel_time', 'coefficient': 0.7},
                                         {'weight': 'length', 'coefficient': 0.3}
                                     ],
                                     vehicle_type: str = 'van') -> Dict:
        """
        Find optimal route considering multiple objectives and vehicle constraints
        
        Args:
            origin: (lat, lon) start point
            destination: (lat, lon) end point
            objectives: List of optimization objectives with weights
            vehicle_type: Vehicle profile to apply
            
        Returns:
            Dictionary with route details and metrics
        """
        self._validate_vehicle_profile(vehicle_type)
        vehicle_profile = self._vehicle_profiles[vehicle_type]
        
        # Apply vehicle constraints
        constrained_graph = self._apply_vehicle_constraints(
            self.graph.copy(), 
            vehicle_profile
        )
        
        # Create composite weight
        for _, _, data in constrained_graph.edges(data=True):
            data['composite_weight'] = sum(
                data[obj['weight']] * obj['coefficient']
                for obj in objectives
            )
        
        # Find optimal path
        orig_node = ox.nearest_nodes(constrained_graph, origin[1], origin[0])
        dest_node = ox.nearest_nodes(constrained_graph, destination[1], destination[0])
        
        try:
            path = nx.shortest_path(
                constrained_graph, 
                orig_node, 
                dest_node, 
                weight='composite_weight'
            )
        except nx.NetworkXNoPath:
            logger.error("No viable path found with current constraints")
            return None
            
        return self._compile_route_details(constrained_graph, path, vehicle_profile)

    def _apply_vehicle_constraints(self, 
                                  graph: nx.MultiDiGraph,
                                  profile: Dict) -> nx.MultiDiGraph:
        """Modify network based on vehicle limitations"""
        for u, v, k, data in graph.edges(keys=True, data=True):
            # Apply height restrictions
            if 'maxheight' in data and profile.get('height'):
                if data['maxheight'] < profile['height']:
                    data['restricted'] = True
                    
            # Apply weight restrictions
            if 'maxweight' in data and profile.get('max_weight'):
                if data['maxweight'] < profile['max_weight']:
                    data['restricted'] = True
                    
            # Apply hazardous material restrictions
            if profile.get('hazardous'):
                if data.get('hazmat_prohibited', False):
                    data['restricted'] = True
                    
            # Adjust speed based on vehicle profile
            if 'speed' in data:
                data['original_speed'] = data['speed']
                data['speed'] *= profile.get('speed_factor', 1.0)
                data['travel_time'] = data['length'] / (data['speed'] * 0.277778)  # m/s
                
            # Apply grade limitations
            if 'grade' in data and profile.get('max_grade'):
                if abs(data['grade']) > profile['max_grade']:
                    data['restricted'] = True
                    
        # Remove restricted edges
        remove_edges = [(u, v, k) for u, v, k, data in graph.edges(keys=True, data=True)
                       if data.get('restricted', False)]
        graph.remove_edges_from(remove_edges)
        
        return graph

    def _compile_route_details(self, 
                             graph: nx.MultiDiGraph, 
                             path: List[int],
                             profile: Dict) -> Dict:
        """Generate detailed route metrics"""
        route_edges = ox.utils_graph.get_route_edge_attributes(graph, path)
        
        total_length = sum(e['length'] for e in route_edges)
        total_time = sum(e['travel_time'] for e in route_edges)
        elevation_gain = sum(max(0, e['grade'] * e['length']) for e in route_edges)
        
        return {
            'geometry': self._path_to_linestring(graph, path),
            'profile': profile,
            'metrics': {
                'total_km': round(total_length / 1000, 2),
                'estimated_time': round(total_time / 60, 1),  # minutes
                'elevation_gain': round(elevation_gain, 1),
                'num_restrictions': len([e for e in route_edges if e.get('restricted', False)])
            },
            'nodes': path
        }

    def _path_to_linestring(self, graph: nx.MultiDiGraph, path: List[int]) -> LineString:
        """Convert node path to LineString geometry"""
        coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in path]
        return LineString(coords)

    def calculate_service_areas(self, 
                               locations: List[Tuple[float, float]],
                               travel_times: List[int],
                               vehicle_type: str = 'van') -> gpd.GeoDataFrame:
        """
        Calculate isochrones for multiple locations with parallel processing
        
        Args:
            locations: List of (lat, lon) tuples
            travel_times: List of minutes to calculate isochrones for
            vehicle_type: Vehicle profile to use
            
        Returns:
            GeoDataFrame with isochrone polygons
        """
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                self._single_isochrone, loc, travel_times, vehicle_type
            ) for loc in locations]
            
            results = [f.result() for f in futures]
            
        return gpd.GeoDataFrame(pd.concat(results), crs="EPSG:4326")

    def _single_isochrone(self, 
                        location: Tuple[float, float],
                        travel_times: List[int],
                        vehicle_type: str) -> gpd.GeoDataFrame:
        """Calculate isochrones for a single location"""
        constrained_graph = self._apply_vehicle_constraints(
            self.graph.copy(),
            self._vehicle_profiles[vehicle_type]
        )
        
        center_node = ox.nearest_nodes(constrained_graph, location[1], location[0])
        travel_times = sorted(travel_times, reverse=True)
        
        isochrones = []
        for time in travel_times:
            subgraph = nx.ego_graph(
                constrained_graph,
                center_node,
                radius=time*60,  # Convert minutes to seconds
                distance='travel_time',
                undirected=True
            )
            
            if len(subgraph.nodes) == 0:
                continue
                
            points = [Point((constrained_graph.nodes[n]['x'], 
                           constrained_graph.nodes[n]['y'])) 
                     for n in subgraph.nodes]
            
            # Create concave hull
            concave_hull = ox.utils_geo.alpha_shape(points, alpha=0.01)
            if not isinstance(concave_hull, Polygon):
                concave_hull = concave_hull.convex_hull
                
            isochrones.append({
                'location': location,
                'travel_time': time,
                'geometry': concave_hull
            })
            
        return gpd.GeoDataFrame(isochrones)

    def cluster_delivery_locations(self, 
                                  points: List[Tuple[float, float]],
                                  eps_km: float = 5) -> Dict:
        """
        Cluster delivery locations using DBSCAN
        
        Args:
            points: List of (lat, lon) delivery locations
            eps_km: Maximum distance between points in cluster
            
        Returns:
            Dictionary with clusters and centroids
        """
        # Convert to meters for haversine metric
        eps = eps_km * 1000
        
        # Convert to radians for haversine
        coords = np.radians([[p[0], p[1]] for p in points])
        
        db = DBSCAN(
            eps=eps / 6371000,  # Convert meters to radians
            min_samples=1,
            metric='haversine'
        ).fit(coords)
        
        clusters = {}
        for label in set(db.labels_):
            if label == -1:
                continue
                
            cluster_points = [points[i] for i, l in enumerate(db.labels_) if l == label]
            centroid = self._calculate_geometric_median(cluster_points)
            
            clusters[label] = {
                'points': cluster_points,
                'centroid': centroid,
                'count': len(cluster_points)
            }
            
        return clusters

    def _calculate_geometric_median(self, 
                                   points: List[Tuple[float, float]],
                                   eps: float = 1e-5) -> Tuple[float, float]:
        """Compute geometric median using Weiszfeld's algorithm"""
        points = np.array(points)
        median = np.mean(points, axis=0)
        
        for _ in range(100):
            distances = np.linalg.norm(points - median, axis=1)
            distances = np.where(distances < eps, eps, distances)
            weights = 1 / distances
            new_median = np.dot(weights, points) / np.sum(weights)
            
            if np.linalg.norm(new_median - median) < eps:
                break
                
            median = new_median
            
        return (median[0], median[1])

    def _validate_vehicle_profile(self, vehicle_type: str) -> None:
        """Check if vehicle profile exists"""
        if vehicle_type not in self._vehicle_profiles:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}. Available: {list(self._vehicle_profiles.keys())}")

    def visualize_network(self, 
                         graph: Optional[nx.MultiDiGraph] = None,
                         **kwargs) -> folium.Map:
        """Create interactive Folium map of the network"""
        graph = graph or self.graph
        return ox.plot_graph_folium(
            graph,
            graph_map=None,
            popup_attribute='name',
            tiles='CartoDB positron',
            color='#555555',
            **kwargs
        )

def analyze_network_connectivity(graph: nx.MultiDiGraph) -> Dict:
    """Perform advanced network connectivity analysis"""
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(graph, weight='travel_time')
    
    # Find articulation points
    undirected_graph = graph.to_undirected()
    articulation_points = list(nx.articulation_points(undirected_graph))
    
    return {
        'critical_nodes': {
            'max_betweenness': max(betweenness, key=betweenness.get),
            'articulation_points_count': len(articulation_points)
        },
        'connectivity_metrics': {
            'edge_connectivity': nx.edge_connectivity(undirected_graph),
            'node_connectivity': nx.node_connectivity(undirected_graph)
        }
    }

# Example Usage
if __name__ == "__main__":
    logger.add("logs/logistics_network.log")
    
    # Initialize network for Mumbai with truck restrictions
    network = AdvancedLogisticsNetwork(
        network_type='drive',
        custom_filters='["highway"~"motorway|trunk|primary|secondary|tertiary"]'
    )
    network.load_network("Mumbai, India", dist=20000)
    
    # Optimize route for heavy truck
    route = network.optimize_multiobjective_route(
        origin=(19.0760, 72.8777),  # Mumbai coordinates
        destination=(18.5204, 73.8567),  # Pune coordinates
        vehicle_type='heavy_truck'
    )
    
    # Generate service areas
    warehouses = [(19.0760, 72.8777), (18.5204, 73.8567)]
    isochrones = network.calculate_service_areas(
        locations=warehouses,
        travel_times=[30, 60, 90],
        vehicle_type='heavy_truck'
    )
    
    # Visualize results
    m = network.visualize_network()
    folium.GeoJson(isochrones).add_to(m)
    m.save("logistics_network.html")
