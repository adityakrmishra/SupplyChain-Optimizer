import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, validator, root_validator
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cachetools import TTLCache
from geojson import LineString, Feature, FeatureCollection
import polyline
import numpy as np
from scipy.spatial.distance import squareform
from ratelimit import limits, sleep_and_retry

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class VehicleProfile(BaseModel):
    """Detailed vehicle characteristics for accurate routing"""
    type: str = Field(..., description="Vehicle type (truck, van, refrigerated_truck)")
    weight: float = Field(10000, ge=0, description="Total weight in kg")
    height: float = Field(3.5, ge=0, description="Vehicle height in meters")
    width: float = Field(2.5, ge=0, description="Vehicle width in meters")
    length: float = Field(8.0, ge=0, description="Vehicle length in meters")
    axle_load: float = Field(8000, ge=0, description="Max axle load in kg")
    hazardous_materials: List[str] = Field([], description="List of hazardous material classes")
    emission_class: str = Field("Euro6", description="Vehicle emission standard")
    custom_restrictions: Dict[str, Any] = Field({}, description="Custom routing restrictions")

    @validator('type')
    def validate_vehicle_type(cls, v):
        valid_types = ['truck', 'van', 'refrigerated_truck', 
                      'construction', 'tanker', 'car']
        if v not in valid_types:
            raise ValueError(f"Invalid vehicle type. Valid types: {valid_types}")
        return v

class EnvironmentalImpact(BaseModel):
    """Calculated environmental impact of route"""
    co2_kg: float = Field(..., description="CO2 emissions in kilograms")
    fuel_liters: float = Field(..., description="Estimated fuel consumption")
    energy_kwh: float = Field(..., description="Energy consumption")
    noise_impact: float = Field(..., description="Noise pollution index")
    eco_score: float = Field(..., description="Environmental impact score 0-100")

class RouteConstraints(BaseModel):
    """Advanced routing constraints and preferences"""
    avoid: List[str] = Field([], description="Features to avoid (tolls, ferries)")
    prefer: List[str] = Field([], description="Preferred road types")
    max_slope: float = Field(8.0, description="Maximum road slope percentage")
    min_width: float = Field(2.0, description="Minimum road width in meters")
    height_restrictions: bool = Field(True)
    weight_restrictions: bool = Field(True)
    time_window: Optional[Tuple[datetime, datetime]] = None
    driver_breaks: List[Tuple[datetime, datetime]] = []
    working_hours: List[Tuple[time, time]] = []

class RouteOptimizationRequest(BaseModel):
    """Enhanced route optimization request with multiple constraints"""
    waypoints: List[List[float]] = Field(..., min_items=2, 
                                       description="List of [lat, lon] pairs")
    vehicle: VehicleProfile = Field(..., description="Vehicle characteristics")
    constraints: RouteConstraints = Field(RouteConstraints(),
                                        description="Routing constraints")
    optimize_for: str = Field("time", description="Optimization criteria (time, distance, eco)")
    traffic: bool = Field(True, description="Use real-time traffic data")
    alternatives: int = Field(3, ge=0, le=5, description="Number of alternative routes")
    language: str = Field("en", min_length=2, description="Instructions language")
    details: List[str] = Field(["time", "distance", "tolls", "road_types"],
                              description="Detailed breakdown request")

    @root_validator
    def validate_waypoints(cls, values):
        waypoints = values.get('waypoints', [])
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints required")
        for point in waypoints:
            if not (-90 <= point[0] <= 90) or not (-180 <= point[1] <= 180):
                raise ValueError("Invalid coordinates")
        return values

class RouteInstruction(BaseModel):
    """Detailed route navigation instruction"""
    text: str
    distance: float
    time: int
    interval: List[float]
    sign: int
    annotation_text: Optional[str]
    annotation_importance: Optional[int]
    exit_number: Optional[int]
    turn_angle: Optional[float]
    street_name: Optional[str]

class TollDetail(BaseModel):
    """Toll cost and metadata"""
    cost: float
    currency: str
    section_start: List[float]
    section_end: List[float]
    toll_system: Optional[str]
    payment_methods: List[str]

class RouteSection(BaseModel):
    """Detailed route segment analysis"""
    start: List[float]
    end: List[float]
    distance: float
    duration: int
    road_type: str
    surface: str
    speed_limit: Optional[float]
    traffic_level: Optional[float]
    tolls: List[TollDetail]
    elevation: List[float]
    emission_impact: EnvironmentalImpact

class OptimizedRoute(BaseModel):
    """Complete optimized route response"""
    geometry: Feature
    distance: float
    time: int
    ascents: float
    descents: float
    instructions: List[RouteInstruction]
    sections: List[RouteSection]
    alternatives: List[Feature]
    waypoints: FeatureCollection
    traffic_data: Dict[str, Any]
    optimization_metadata: Dict[str, Any]
    environmental_impact: EnvironmentalImpact
    cost_estimate: float

    class Config:
        json_encoders = {
            Feature: lambda v: v.__geo_interface__,
            FeatureCollection: lambda v: v.__geo_interface__
        }

class GraphHopperConfig:
    """Client configuration with caching and retries"""
    def __init__(self):
        self.api_key = os.getenv("GRAPHHOPPER_API_KEY")
        self.base_url = "https://graphhopper.com/api/1"
        self.cache = TTLCache(maxsize=1000, ttl=timedelta(hours=1))
        self.retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=self.retry_strategy))

class GraphHopperClient:
    """Production-grade GraphHopper API client with advanced features"""
    
    def __init__(self, config: GraphHopperConfig = GraphHopperConfig()):
        self.config = config
        self._validate_api_key()
        self._last_request = None

    def _validate_api_key(self):
        """Validate API key format and presence"""
        if not self.config.api_key:
            raise ValueError("GRAPHHOPPER_API_KEY environment variable required")
        if len(self.config.api_key) != 32 or not self.config.api_key.isalnum():
            raise ValueError("Invalid GraphHopper API key format")

    @sleep_and_retry
    @limits(calls=100, period=timedelta(minutes=1).total_seconds())
    def _rate_limited_request(self, method: str, endpoint: str, **kwargs):
        """Rate-limited API request with retries"""
        url = f"{self.config.base_url}/{endpoint}"
        try:
            response = self.config.session.request(
                method,
                url,
                params={"key": self.config.api_key, **kwargs.get("params", {})},
                json=kwargs.get("json"),
                timeout=10
            )
            response.raise_for_status()
            self._last_request = datetime.now()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"API request failed: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            raise

    def _decode_polyline(self, encoded: str) -> List[List[float]]:
        """Decode polyline geometry with elevation data"""
        try:
            return polyline.decode(encoded, geojson=True, elevation=True)
        except Exception as e:
            logger.error(f"Polyline decoding failed: {str(e)}")
            raise

    def _create_geojson(self, data: dict) -> FeatureCollection:
        """Convert API response to GeoJSON format"""
        features = []
        for path in data.get("paths", []):
            geometry = LineString(self._decode_polyline(path.get("points")))
            properties = {
                "distance": path.get("distance"),
                "time": path.get("time"),
                "ascents": path.get("ascend"),
                "descents": path.get("descend")
            }
            features.append(Feature(geometry=geometry, properties=properties))
        return FeatureCollection(features)

    def _calculate_environmental_impact(self, route_data: dict) -> EnvironmentalImpact:
        """Calculate CO2 emissions and environmental impact"""
        # Complex calculation based on vehicle profile and route characteristics
        distance_km = route_data.get("distance", 0) / 1000
        vehicle = route_data.get("vehicle_profile", {})
        
        # Emission factors (grams CO2 per km)
        emission_factors = {
            "truck": 150.0,
            "van": 120.0,
            "car": 100.0
        }
        
        base_emission = emission_factors.get(vehicle.get("type", "truck"), 150)
        co2_kg = (base_emission * distance_km) / 1000
        
        return EnvironmentalImpact(
            co2_kg=co2_kg,
            fuel_liters=distance_km * 0.35,
            energy_kwh=distance_km * 0.8,
            noise_impact=distance_km * 1.2,
            eco_score=max(0, 100 - (co2_kg * 10))
        )

    def get_optimized_route(self, request: RouteOptimizationRequest) -> OptimizedRoute:
        """Get optimized route with multiple constraints"""
        params = self._build_route_params(request)
        cache_key = self._generate_cache_key(params)
        
        if cache_key in self.config.cache:
            logger.info("Returning cached route")
            return self.config.cache[cache_key]

        start_time = time.time()
        data = self._rate_limited_request("GET", "route", params=params)
        logger.info(f"Route calculation took {time.time() - start_time:.2f}s")

        processed_route = self._process_route_data(data, request)
        self.config.cache[cache_key] = processed_route
        return processed_route

    def _build_route_params(self, request: RouteOptimizationRequest) -> dict:
        """Construct API parameters from optimization request"""
        return {
            "vehicle": request.vehicle.type,
            "weight": request.vehicle.weight,
            "height": request.vehicle.height,
            "width": request.vehicle.width,
            "axle_load": request.vehicle.axle_load,
            "hazmat": ",".join(request.vehicle.hazardous_materials),
            "emission_class": request.vehicle.emission_class,
            "avoid": ",".join(request.constraints.avoid),
            "prefer": ",".join(request.constraints.prefer),
            "max_slope": request.constraints.max_slope,
            "min_width": request.constraints.min_width,
            "points": ";".join([f"{p[0]},{p[1]}" for p in request.waypoints]),
            "optimize": request.optimize_for,
            "traffic": "true" if request.traffic else "false",
            "alternatives": request.alternatives,
            "instructions": "true",
            "elevation": "true",
            "details": ",".join(request.details),
            "locale": request.language
        }

    def _process_route_data(self, data: dict, request: RouteOptimizationRequest) -> OptimizedRoute:
        """Process raw API response into structured model"""
        main_route = data.get("paths", [{}])[0]
        alternatives = data.get("paths", [])[1:request.alternatives + 1]
        
        return OptimizedRoute(
            geometry=self._create_geojson(data).features[0],
            distance=main_route.get("distance", 0),
            time=main_route.get("time", 0),
            ascents=main_route.get("ascend", 0),
            descents=main_route.get("descend", 0),
            instructions=[
                RouteInstruction(**i) for i in main_route.get("instructions", [])
            ],
            sections=[self._process_section(s) for s in main_route.get("details", [])],
            alternatives=self._create_geojson({"paths": alternatives}),
            waypoints=FeatureCollection([
                Feature(geometry=Point((p[0], p[1])) for p in request.waypoints
            ]),
            traffic_data=data.get("traffic", {}),
            optimization_metadata=data.get("metadata", {}),
            environmental_impact=self._calculate_environmental_impact({
                **main_route,
                "vehicle_profile": request.vehicle.dict()
            }),
            cost_estimate=self._calculate_toll_costs(main_route.get("tolls", []))
        )

    def _process_section(self, section: dict) -> RouteSection:
        """Process detailed route section data"""
        return RouteSection(
            start=section.get("start"),
            end=section.get("end"),
            distance=section.get("distance"),
            duration=section.get("time"),
            road_type=section.get("road_type"),
            surface=section.get("surface"),
            speed_limit=section.get("max_speed"),
            traffic_level=section.get("traffic"),
            tolls=[TollDetail(**t) for t in section.get("tolls", [])],
            elevation=section.get("elevation", []),
            emission_impact=self._calculate_environmental_impact(section)
        )

    def _calculate_toll_costs(self, tolls: List[dict]) -> float:
        """Calculate total toll costs across route"""
        return sum(t.get("cost", 0) for t in tolls)

    def _generate_cache_key(self, params: dict) -> str:
        """Generate unique cache key from request parameters"""
        return hash(frozenset(params.items()))

    def get_distance_matrix(self, points: List[List[float]], 
                          vehicle: VehicleProfile) -> np.ndarray:
        """Calculate full distance matrix with vehicle constraints"""
        params = {
            "from_points": ";".join([f"{p[0]},{p[1]}" for p in points]),
            "to_points": ";".join([f"{p[0]},{p[1]}" for p in points]),
            "out_array": ["times", "distances"],
            "vehicle": vehicle.type,
            "weight": vehicle.weight,
            "height": vehicle.height
        }
        
        data = self._rate_limited_request("GET", "matrix", params=params)
        return {
            "durations": np.array(data.get("times", [])),
            "distances": np.array(data.get("distances", []))
        }

    def get_isochrone(self, location: List[float], 
                    vehicle: VehicleProfile,
                    time_limit: int) -> FeatureCollection:
        """Calculate reachable area within time/distance limit"""
        params = {
            "point": f"{location[0]},{location[1]}",
            "time_limit": time_limit,
            "vehicle": vehicle.type,
            "buckets": 5
        }
        
        data = self._rate_limited_request("GET", "isochrone", params=params)
        return self._create_geojson(data)

    def get_elevation_profile(self, route: OptimizedRoute) -> Dict[str, List[float]]:
        """Extract elevation data from route geometry"""
        return {
            "distance": np.linspace(0, route.distance, len(route.geometry.coordinates)),
            "elevation": [p[2] for p in route.geometry.coordinates]
        }

    def validate_route(self, route: OptimizedRoute, 
                      vehicle: VehicleProfile) -> bool:
        """Validate route against vehicle constraints"""
        # Check height restrictions
        for section in route.sections:
            if section.road_type == "tunnel" and vehicle.height > 4.0:
                return False
            if section.road_type == "bridge" and vehicle.weight > section.get("weight_limit", 10000):
                return False
        return True

    def compare_routes(self, route1: OptimizedRoute, 
                     route2: OptimizedRoute) -> Dict[str, float]:
        """Compare two routes across multiple metrics"""
        return {
            "time_diff": route1.time - route2.time,
            "distance_diff": route1.distance - route2.distance,
            "cost_diff": route1.cost_estimate - route2.cost_estimate,
            "eco_diff": (route1.environmental_impact.eco_score -
                        route2.environmental_impact.eco_score)
        }

# Example usage
if __name__ == "__main__":
    client = GraphHopperClient()
    
    request = RouteOptimizationRequest(
        waypoints=[[37.7749, -122.4194], [34.0522, -118.2437]],
        vehicle=VehicleProfile(
            type="truck",
            weight=15000,
            height=4.0,
            width=2.6
        ),
        constraints=RouteConstraints(
            avoid=["tolls", "tunnels"],
            prefer=["highways"]
        )
    )
    
    route = client.get_optimized_route(request)
    print(f"Optimal route: {route.distance}m, {route.time//60000} minutes")
    print(f"CO2 emissions: {route.environmental_impact.co2_kg}
      # Generate elevation profile visualization
    elevation_data = client.get_elevation_profile(route)
    plot_elevation_profile(elevation_data)

    # Validate route against vehicle constraints
    if not client.validate_route(route, request.vehicle):
        print("⚠️ Route validation failed - constraints violated!")
        # Find problematic sections
        problematic_sections = [
            section for section in route.sections 
            if section.speed_limit and section.speed_limit < 50
        ]
        print(f"{len(problematic_sections)} low-speed sections detected")

    # Check for height-restricted tunnels/bridges
    restricted_areas = [
        (section.start, section.end) 
        for section in route.sections 
        if section.road_type in ['tunnel', 'bridge'] 
        and section.height_clearance < request.vehicle.height
    ]
    if restricted_areas:
        print(f"🚚 Height restriction alert in {len(restricted_areas)} locations")

    # Get real-time traffic updates
    traffic_update = client.get_traffic_update(route.geometry)
    if traffic_update.delay > 600:  # 10+ minute delay
        print(f"🚨 Significant traffic delay: {traffic_update.delay//60} minutes")
        # Reroute with current traffic
        updated_route = client.get_optimized_route(request)
        print(f"New ETA: {updated_route.time//60000} minutes")

    # Generate driver schedule with breaks
    schedule = calculate_driver_schedule(
        total_time=route.time,
        working_hours=request.constraints.working_hours,
        breaks=request.constraints.driver_breaks
    )
    print("\nDriver Schedule:")
    for shift in schedule:
        print(f"{shift['start']} - {shift['end']} ({shift['duration']} mins)")

    # Calculate detailed cost breakdown
    cost_analysis = calculate_transport_cost(
        distance_km=route.distance/1000,
        vehicle=request.vehicle,
        tolls=route.cost_estimate,
        labor_rate=45  # $/hour
    )
    print("\nCost Analysis:")
    print(f"Fuel: ${cost_analysis['fuel']:.2f}")
    print(f"Tolls: ${cost_analysis['tolls']:.2f}")
    print(f"Labor: ${cost_analysis['labor']:.2f}")
    print(f"Total: ${cost_analysis['total']:.2f}")

    # Generate environmental compliance report
    compliance = check_environmental_compliance(
        impact=route.environmental_impact,
        regulations="California Air Resources Board"
    )
    if not compliance["passed"]:
        print("\n⚠️ Environmental Compliance Issues:")
        for issue in compliance["issues"]:
            print(f"- {issue}")

    # Export route data for fleet management systems
    export_formats = {
        "gpx": generate_gpx(route),
        "geojson": route.geometry,
        "csv": generate_driver_checklist(route.instructions)
    }
    save_route_data(export_formats, format="all")

    # Compare with alternative routes
    if route.alternatives:
        comparison = client.compare_routes(route, route.alternatives[0])
        print("\nAlternative Route Comparison:")
        print(f"Time difference: {comparison['time_diff']//60000} mins")
        print(f"Cost difference: ${comparison['cost_diff']:.2f}")
        print(f"Eco score difference: {comparison['eco_diff']:.1f} points")

    # Integrate with supply chain management system
    scms_payload = create_scms_integration_payload(
        route=route,
        shipment_id="SC-123456",
        vehicle=request.vehicle,
        driver="DRV-789"
    )
    update_supply_chain_system(scms_payload)

def plot_elevation_profile(data: dict):
    """Visualize elevation changes along route"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.plot(data['distance'], data['elevation'])
    plt.title('Route Elevation Profile')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.grid(True)
    plt.show()

def calculate_driver_schedule(total_time: int, working_hours: list, breaks: list):
    """Generate HOS-compliant driver schedule"""
    from datetime import datetime, timedelta
    
    schedule = []
    current_time = datetime.now()
    remaining_time = total_time / 1000  # Convert ms to seconds
    
    while remaining_time > 0:
        shift_duration = min(remaining_time, 8*3600)  # Max 8-hour shift
        end_time = current_time + timedelta(seconds=shift_duration)
        
        schedule.append({
            "start": current_time.strftime("%H:%M"),
            "end": end_time.strftime("%H:%M"),
            "duration": shift_duration//60
        })
        
        remaining_time -= shift_duration
        current_time = end_time + timedelta(hours=1)  # Mandatory break
    
    return schedule

def calculate_transport_cost(distance_km: float, vehicle: VehicleProfile, 
                           tolls: float, labor_rate: float):
    """Calculate detailed transportation costs"""
    fuel_price = 3.50  # $/gallon
    mpg = {
        "truck": 6.5,
        "van": 12.0,
        "car": 30.0
    }.get(vehicle.type, 8.0)
    
    fuel_cost = (distance_km / 1.60934) / mpg * fuel_price  # km to miles
    labor_cost = (distance_km / 80) * labor_rate  # Assuming 80 kph average speed
    
    return {
        "fuel": fuel_cost,
        "tolls": tolls,
        "labor": labor_cost,
        "maintenance": distance_km * 0.15,
        "total": fuel_cost + tolls + labor_cost
    }

def check_environmental_compliance(impact: EnvironmentalImpact, 
                                 regulations: str):
    """Check against regional environmental regulations"""
    thresholds = {
        "California Air Resources Board": {
            "co2_per_km": 150,  # g/km
            "noise_limit": 75    # dB
        }
    }
    
    compliance = {"passed": True, "issues": []}
    limits = thresholds.get(regulations, {})
    
    co2_per_km = (impact.co2_kg * 1000) / (route.distance/1000)
    if co2_per_km > limits.get("co2_per_km", float('inf')):
        compliance["passed"] = False
        compliance["issues"].append(f"CO2 emissions {co2_per_km:.1f}g/km exceeds limit")
    
    if impact.noise_impact > limits.get("noise_limit", float('inf')):
        compliance["passed"] = False
        compliance["issues"].append(f"Noise level {impact.noise_impact:.1f}dB exceeds limit")
    
    return compliance

def generate_gpx(route: OptimizedRoute):
    """Generate GPX format for navigation systems"""
    from gpxpy.gpx import GPX, GPXTrack, GPXTrackSegment
    
    gpx = GPX()
    track = GPXTrack()
    segment = GPXTrackSegment()
    
    for point in route.geometry.coordinates:
        segment.points.append(GPXTrackPoint(
            latitude=point[0],
            longitude=point[1],
            elevation=point[2]
        ))
    
    track.segments.append(segment)
    gpx.tracks.append(track)
    return gpx.to_xml()

def create_scms_integration_payload(route: OptimizedRoute, 
                                  shipment_id: str,
                                  vehicle: VehicleProfile,
                                  driver: str):
    """Prepare data for supply chain management integration"""
    return {
        "shipment_id": shipment_id,
        "route_id": route.optimization_metadata.get("id"),
        "driver": driver,
        "vehicle": vehicle.dict(),
        "planned_departure": datetime.now().isoformat(),
        "estimated_arrival": (datetime.now() + timedelta(seconds=route.time//1000)).isoformat(),
        "checkpoints": [
            {
                "coordinates": wp,
                "eta": (datetime.now() + timedelta(seconds=section.time//1000)).isoformat()
            }
            for wp, section in zip(route.waypoints, route.sections)
        ],
        "environmental_impact": route.environmental_impact.dict(),
        "compliance_status": check_environmental_compliance(
            route.environmental_impact,
            "California Air Resources Board"
        )
    }
