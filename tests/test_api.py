import pytest
from fastapi.testclient import TestClient
from api.main import app
from blockchain.web3_integration import BlockchainIntegrator

client = TestClient(app)

@patch('blockchain.web3_integration.BlockchainIntegrator')
def test_create_shipment(mock_blockchain):
    mock_blockchain.return_value.update_shipment_status.return_value = {"tx_hash": "0x123"}
    
    response = client.post(
        "/shipments",
        json={
            "origin": "40.7128,-74.0060",
            "destination": "34.0522,-118.2437",
            "items": ["electronics"]
        },
        headers={"Authorization": "Bearer test"}
    )
    
    assert response.status_code == 201
    assert "blockchain_tx" in response.json()

def test_route_optimization():
    response = client.post(
        "/routes/optimize",
        json={
            "waypoints": ["51.5074,-0.1278", "48.8566,2.3522"],
            "constraints": {"max_co2": 200}
        }
    )
    
    assert response.status_code == 200
    assert "optimal_route" in response.json()
    assert "alternatives" in response.json()

def test_auth_failure():
    response = client.get("/shipments/SHIP-123")
    assert response.status_code == 401
