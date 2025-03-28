import pytest
from route_optimizer.graphhopper_api import GraphHopperClient, RouteOptimizationRequest
from unittest.mock import patch

@pytest.fixture
def mock_gh_client():
    with patch('requests.get') as mock_get:
        client = GraphHopperClient()
        mock_response = {
            "paths": [{
                "distance": 1000,
                "time": 3600,
                "points": "encoded_string"
            }]
        }
        mock_get.return_value.json.return_value = mock_response
        yield client

def test_route_optimization(mock_gh_client):
    request = RouteOptimizationRequest(
        waypoints=[[52.5200, 13.4050], [48.8566, 2.3522]]
    )
    route = mock_gh_client.get_route(request)
    
    assert route.distance == 1000
    assert route.time == 3600
    assert route.points == "encoded_string"

def test_carbon_calculation():
    from route_optimizer.carbon_footprint import CarbonCalculator
    calc = CarbonCalculator()
    
    emissions = calc.calculate_co2(
        distance_km=100,
        vehicle_type="diesel_truck",
        load_kg=5000,
        empty_return=True
    )
    
    assert pytest.approx(emissions, 0.162*100*5*2)  # Base rate * km * tons * round trip
