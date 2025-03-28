import pytest
from web3 import Web3
from blockchain.web3_integration import BlockchainIntegrator
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_web3():
    web3 = MagicMock(spec=Web3)
    web3.eth = MagicMock()
    web3.eth.contract = MagicMock()
    return web3

@pytest.fixture
def blockchain_integrator(mock_web3):
    with patch('blockchain.web3_integration.Web3', return_value=mock_web3):
        integrator = BlockchainIntegrator()
        integrator.contract = MagicMock()
        return integrator

def test_shipment_update(blockchain_integrator):
    tx_receipt = blockchain_integrator.update_shipment_status(
        "SHIP-123", "IN_TRANSIT", "NYC", "test_private_key"
    )
    
    blockchain_integrator.contract.functions.updateShipmentStatus.assert_called_once_with(
        "SHIP-123", "IN_TRANSIT", "NYC"
    )
    assert isinstance(tx_receipt, dict)

def test_invalid_shipment(blockchain_integrator):
    blockchain_integrator.contract.functions.shipments.side_effect = Exception("Not found")
    
    with pytest.raises(RuntimeError) as excinfo:
        blockchain_integrator.get_shipment_details("INVALID-ID")
    
    assert "Error fetching shipment" in str(excinfo.value)

def test_event_listening(blockchain_integrator):
    mock_event = MagicMock()
    mock_event.get.return_value = {'args': {
        'shipmentId': 'SHIP-123',
        'status': 'DELIVERED'
    }}
    
    blockchain_integrator.contract.events.ShipmentUpdated.create_filter = MagicMock(
        return_value=MagicMock(get_all_entries=lambda: [mock_event])
    )
    
    events = blockchain_integrator.listen_for_events('ShipmentUpdated', 10)
    assert len(events) == 1
    assert events[0]['args']['shipmentId'] == 'SHIP-123'

def test_gas_estimation(blockchain_integrator):
    blockchain_integrator.contract.functions.updateShipmentStatus.estimate_gas.return_value = 21000
    gas_estimate = blockchain_integrator.estimate_gas(
        "updateShipmentStatus", ["SHIP-123", "DELAYED", "CHI"]
    )
    
    assert gas_estimate == 21000
    blockchain_integrator.contract.functions.updateShipmentStatus.assert_called_once_with(
        "SHIP-123", "DELAYED", "CHI"
    )
