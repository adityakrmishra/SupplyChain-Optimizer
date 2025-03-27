import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from web3.contract import Contract
from web3.datastructures import AttributeDict
from web3.exceptions import ContractLogicError
from web3.middleware import geth_poa_middleware
from web3.types import TxReceipt
from eth_typing import ChecksumAddress
from cryptography.fernet import Fernet
from geoalchemy2 import WKBElement
from geoalchemy2.shape import to_shape
from .schemas import (
    ShipmentCreate,
    ShipmentUpdate,
    ShipmentStatus,
    Location,
    ShipmentEvent
)

class BlockchainError(Exception):
    """Base exception for blockchain operations"""
    pass

class ContractDeploymentError(BlockchainError):
    pass

class ShipmentUpdateError(BlockchainError):
    pass

class BlockchainIntegrator:
    def __init__(self, network: str = "ethereum"):
        """
        Initialize blockchain integrator with network configuration
        
        Args:
            network: Blockchain network (ethereum|polygon|avalanche)
        """
        self.network = network
        self._configure_network()
        self.w3 = Web3(Web3.HTTPProvider(os.getenv("BLOCKCHAIN_NODE_URL")))
        
        if self.network in ["polygon", "avalanche"]:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
        self.contracts: Dict[str, Contract] = {}
        self._load_contract_config()
        self._decryption_key = Fernet(os.getenv("ENCRYPTION_KEY"))

    def _configure_network(self) -> None:
        """Set network-specific parameters"""
        self.network_config = {
            "ethereum": {"chain_id": 1, "gas_oracle": "eth_gasStation"},
            "polygon": {"chain_id": 137, "gas_oracle": "polygon_gasStation"},
            "avalanche": {"chain_id": 43114, "gas_oracle": "avax_gasStation"}
        }.get(self.network, {})

    def _load_contract_config(self) -> None:
        """Load ABI and bytecode for all contracts"""
        self.contract_abis = {}
        contracts = ["SupplyChain", "PaymentHandler", "Compliance"]
        
        for contract in contracts:
            with open(f"blockchain/smart_contracts/{contract}.json") as f:
                data = json.load(f)
                self.contract_abis[contract] = {
                    "abi": data["abi"],
                    "bytecode": data["bytecode"]
                }

    def _get_gas_price(self) -> int:
        """Get dynamic gas price based on network conditions"""
        try:
            if "gas_oracle" in self.network_config:
                return self.w3.eth.gas_price
            return self.w3.to_wei("50", "gwei")
        except Exception as e:
            raise BlockchainError(f"Gas price estimation failed: {str(e)}")

    def deploy_contract(self, contract_name: str, private_key: str) -> TxReceipt:
        """
        Deploy a new smart contract to the blockchain
        
        Args:
            contract_name: Name of contract to deploy
            private_key: Encrypted private key for deployment
            
        Returns:
            Transaction receipt with contract address
        """
        try:
            decrypted_key = self._decrypt_private_key(private_key)
            account = self.w3.eth.account.from_key(decrypted_key)
            
            contract_config = self.contract_abis.get(contract_name)
            if not contract_config:
                raise ContractDeploymentError(f"Contract {contract_name} not found")
                
            contract = self.w3.eth.contract(
                abi=contract_config["abi"],
                bytecode=contract_config["bytecode"]
            )
            
            tx = contract.constructor().build_transaction({
                "chainId": self.network_config["chain_id"],
                "gas": 3000000,
                "gasPrice": self._get_gas_price(),
                "nonce": self.w3.eth.get_transaction_count(account.address),
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if not receipt.contractAddress:
                raise ContractDeploymentError("Contract deployment failed")
                
            self.contracts[contract_name] = self.w3.eth.contract(
                address=receipt.contractAddress,
                abi=contract_config["abi"]
            )
            return receipt
            
        except ContractLogicError as e:
            raise ContractDeploymentError(f"Contract logic error: {str(e)}")
        except Exception as e:
            raise ContractDeploymentError(f"Deployment failed: {str(e)}")

    def _get_contract(self, contract_name: str) -> Contract:
        """Get initialized contract instance"""
        contract = self.contracts.get(contract_name)
        if not contract:
            raise BlockchainError(f"Contract {contract_name} not deployed")
        return contract

    def create_shipment(self, shipment: ShipmentCreate, private_key: str) -> TxReceipt:
        """
        Create a new shipment on the blockchain
        
        Args:
            shipment: Shipment creation data
            private_key: Encrypted private key
            
        Returns:
            Transaction receipt
        """
        contract = self._get_contract("SupplyChain")
        decrypted_key = self._decrypt_private_key(private_key)
        account = self.w3.eth.account.from_key(decrypted_key)
        
        try:
            tx = contract.functions.createShipment(
                shipment.id,
                [item.dict() for item in shipment.items],
                shipment.supplier,
                shipment.carrier,
                self._location_to_tuple(shipment.origin)
            ).build_transaction({
                "chainId": self.network_config["chain_id"],
                "gas": 800000,
                "gasPrice": self._get_gas_price(),
                "nonce": self.w3.eth.get_transaction_count(account.address),
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            return self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
        except ContractLogicError as e:
            raise ShipmentUpdateError(f"Shipment creation failed: {str(e)}")

    def update_shipment_status(
        self, 
        shipment_id: str,
        status: ShipmentStatus,
        location: Location,
        private_key: str
    ) -> TxReceipt:
        """
        Update shipment status and location
        
        Args:
            shipment_id: Unique shipment identifier
            status: New status from ShipmentStatus enum
            location: Current GPS coordinates
            private_key: Encrypted private key
            
        Returns:
            Transaction receipt
        """
        contract = self._get_contract("SupplyChain")
        decrypted_key = self._decrypt_private_key(private_key)
        account = self.w3.eth.account.from_key(decrypted_key)
        
        try:
            tx = contract.functions.updateShipmentStatus(
                shipment_id,
                status.value,
                self._location_to_tuple(location)
            ).build_transaction({
                "chainId": self.network_config["chain_id"],
                "gas": 500000,
                "gasPrice": self._get_gas_price(),
                "nonce": self.w3.eth.get_transaction_count(account.address),
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            return self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
        except ContractLogicError as e:
            raise ShipmentUpdateError(f"Status update failed: {str(e)}")

    def get_shipment_history(self, shipment_id: str) -> List[ShipmentEvent]:
        """
        Get complete history of a shipment
        
        Args:
            shipment_id: Unique shipment identifier
            
        Returns:
            List of shipment events with timestamps
        """
        contract = self._get_contract("SupplyChain")
        events = contract.events.ShipmentUpdated.get_logs(
            argument_filters={"shipmentId": shipment_id}
        )
        
        return [
            ShipmentEvent(
                status=event.args.status,
                location=Location(
                    lat=event.args.location[0],
                    lng=event.args.location[1]
                ),
                timestamp=datetime.fromtimestamp(event.args.timestamp),
                block_number=event.blockNumber
            ) for event in events
        ]

    def _location_to_tuple(self, location: Location) -> Tuple[float, float]:
        """Convert Location object to blockchain-friendly tuple"""
        return (location.lat, location.lng)

    def _decrypt_private_key(self, encrypted_key: str) -> str:
        """Decrypt encrypted private key"""
        try:
            return self._decryption_key.decrypt(encrypted_key.encode()).decode()
        except Exception as e:
            raise BlockchainError(f"Key decryption failed: {str(e)}")

    # Additional functionality continues...

    def verify_customs_approval(self, shipment_id: str) -> bool:
        """
        Verify customs clearance status for a shipment
        
        Args:
            shipment_id: Unique shipment identifier
            
        Returns:
            True if shipment has required customs approvals
        """
        contract = self._get_contract("Compliance")
        return contract.functions.verifyCustoms(shipment_id).call()

    def process_payment(
        self,
        shipment_id: str,
        amount: float,
        token_address: Optional[str] = None,
        private_key: str
    ) -> TxReceipt:
        """
        Process payment for shipment using native or token currency
        
        Args:
            shipment_id: Associated shipment ID
            amount: Payment amount
            token_address: ERC20 token address (optional)
            private_key: Encrypted private key
            
        Returns:
            Transaction receipt
        """
        decrypted_key = self._decrypt_private_key(private_key)
        account = self.w3.eth.account.from_key(decrypted_key)
        
        if token_address:
            return self._process_token_payment(
                shipment_id, amount, token_address, account
            )
        else:
            return self._process_native_payment(shipment_id, amount, account)

    def _process_token_payment(self, shipment_id, amount, token_address, account):
        """Handle ERC20 token payment"""
        payment_contract = self._get_contract("PaymentHandler")
        token_contract = self.w3.eth.contract(
            address=token_address,
            abi=self.contract_abis["PaymentHandler"]["abi"]
        )
        
        # Approve token transfer
        approve_tx = token_contract.functions.approve(
            payment_contract.address,
            self.w3.to_wei(amount, "ether")
        ).build_transaction({
            "chainId": self.network_config["chain_id"],
            "gas": 200000,
            "gasPrice": self._get_gas_price(),
            "nonce": self.w3.eth.get_transaction_count(account.address),
        })
        
        signed_approve = account.sign_transaction(approve_tx)
        self.w3.eth.send_raw_transaction(signed_approve.rawTransaction)
        
        # Execute payment
        payment_tx = payment_contract.functions.processTokenPayment(
            shipment_id,
            token_address,
            self.w3.to_wei(amount, "ether")
        ).build_transaction({
            "chainId": self.network_config["chain_id"],
            "gas": 300000,
            "gasPrice": self._get_gas_price(),
            "nonce": self.w3.eth.get_transaction_count(account.address) + 1,
        })
        
        signed_payment = account.sign_transaction(payment_tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_payment.rawTransaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def _process_native_payment(self, shipment_id, amount, account):
        """Handle native cryptocurrency payment"""
        contract = self._get_contract("PaymentHandler")
        tx = contract.functions.processNativePayment(shipment_id).build_transaction({
            "chainId": self.network_config["chain_id"],
            "value": self.w3.to_wei(amount, "ether"),
            "gas": 150000,
            "gasPrice": self._get_gas_price(),
            "nonce": self.w3.eth.get_transaction_count(account.address),
        })
        
        signed_tx = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def get_shipment_temperature_data(self, shipment_id: str) -> List[float]:
        """
        Retrieve temperature logs for a shipment
        
        Args:
            shipment_id: Unique shipment identifier
            
        Returns:
            List of temperature readings in Celsius
        """
        contract = self._get_contract("SupplyChain")
        return contract.functions.getTemperatureData(shipment_id).call()

    def generate_sustainability_report(self, shipment_id: str) -> Dict:
        """
        Generate carbon footprint report for shipment
        
        Args:
            shipment_id: Unique shipment identifier
            
        Returns:
            Dictionary with emissions data and offsets
        """
        contract = self._get_contract("Compliance")
        raw_data = contract.functions.getSustainabilityData(shipment_id).call()
        
        return {
            "co2_emissions": raw_data[0] / 1000,  # Convert to kg
            "distance_traveled": raw_data[1],
            "carbon_offset": raw_data[2] / 1000,
            "transport_modes": raw_data[3]
        }

    def register_iot_device(self, device_id: str, private_key: str) -> TxReceipt:
        """
        Register new IoT device for automated updates
        
        Args:
            device_id: Unique device identifier
            private_key: Encrypted private key
            
        Returns:
            Transaction receipt
        """
        contract = self._get_contract("SupplyChain")
        decrypted_key = self._decrypt_private_key(private_key)
        account = self.w3.eth.account.from_key(decrypted_key)
        
        tx = contract.functions.registerDevice(device_id).build_transaction({
            "chainId": self.network_config["chain_id"],
            "gas": 300000,
            "gasPrice": self._get_gas_price(),
            "nonce": self.w3.eth.get_transaction_count(account.address),
        })
        
        signed_tx = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def listen_for_events(self, event_name: str, timeout: int = 60):
        """
        Listen for real-time blockchain events
        
        Args:
            event_name: Name of event to listen for
            timeout: Timeout in seconds
            
        Yields:
            Parsed event data
        """
        contract = self._get_contract("SupplyChain")
        event_filter = contract.events[event_name].create_filter(fromBlock="latest")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            for event in event_filter.get_new_entries():
                yield self._parse_event(event)
            time.sleep(2)

    def _parse_event(self, event: AttributeDict) -> Dict:
        """Parse raw blockchain event into standardized format"""
        return {
            "event": event.event,
            "block": event.blockNumber,
            "timestamp": datetime.fromtimestamp(
                self.w3.eth.get_block(event.blockNumber).timestamp
            ),
            "data": dict(event.args)
        }

# Add to imports
import hashlib
from typing import Union
from shapely.geometry import Point, shape

class BlockchainIntegrator:
    # ... existing code ...
    
    # ======================
    # Geo-Conversion Methods
    # ======================
    
    def _convert_wkb_to_tuple(self, wkb_element: WKBElement) -> Tuple[float, float]:
        """
        Convert PostGIS WKB geometry to latitude/longitude tuple
        
        Args:
            wkb_element: GeoAlchemy2 WKBElement
            
        Returns:
            (latitude, longitude) tuple
        """
        try:
            point = to_shape(wkb_element)
            return (point.y, point.x)  # PostGIS stores coordinates as (longitude, latitude)
        except Exception as e:
            raise ValueError(f"Invalid WKB element: {str(e)}")

    def _convert_geojson_to_tuple(self, geojson: str) -> Tuple[float, float]:
        """
        Convert GeoJSON point to latitude/longitude tuple
        
        Args:
            geojson: GeoJSON string
            
        Returns:
            (latitude, longitude) tuple
            
        Example input:
            '{"type": "Point", "coordinates": [100.0, 0.0]}'
        """
        try:
            data = json.loads(geojson)
            geometry = shape(data)
            if not isinstance(geometry, Point):
                raise ValueError("Only Point geometry is supported")
            return (geometry.y, geometry.x)
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON: {str(e)}")

    def _location_to_geohash(self, location: Location, precision: int = 9) -> str:
        """
        Convert location to geohash string
        
        Args:
            location: Location object
            precision: Geohash precision (1-12)
            
        Returns:
            Geohash string
        """
        import geohash  # Requires python-geohash package
        return geohash.encode(location.lat, location.lng, precision=precision)

    # =====================
    # Encryption/Decryption
    # =====================
    
    def _encrypt_data(self, data: Union[str, bytes]) -> str:
        """
        Encrypt sensitive data before blockchain storage
        
        Args:
            data: String or bytes to encrypt
            
        Returns:
            Encrypted string (Fernet token)
        """
        if isinstance(data, str):
            data = data.encode()
        return self._decryption_key.encrypt(data).decode()

    def _decrypt_data(self, token: str) -> str:
        """
        Decrypt data from blockchain storage
        
        Args:
            token: Encrypted string from blockchain
            
        Returns:
            Decrypted UTF-8 string
        """
        return self._decryption_key.decrypt(token.encode()).decode()

    def generate_data_hash(self, data: dict) -> str:
        """
        Generate SHA-256 hash of shipment data
        
        Args:
            data: Dictionary of shipment data
            
        Returns:
            Hexadecimal hash string
        """
        serialized = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()

    # =================
    # Validation Methods
    # =================
    
    def _validate_location(self, lat: float, lng: float) -> None:
        """
        Validate geographic coordinates
        
        Args:
            lat: Latitude (-90 to 90)
            lng: Longitude (-180 to 180)
            
        Raises:
            ValueError for invalid coordinates
        """
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90")
        if not (-180 <= lng <= 180):
            raise ValueError(f"Invalid longitude: {lng}. Must be between -180 and 180")

    def _validate_shipment_id(self, shipment_id: str) -> None:
        """
        Validate shipment ID format (EIN-XXXX-XXXX-XXXX)
        
        Args:
            shipment_id: Shipment identifier
            
        Raises:
            ValueError for invalid format
        """
        import re
        pattern = r"^EIN-\d{4}-\d{4}-\d{4}$"
        if not re.match(pattern, shipment_id):
            raise ValueError("Invalid shipment ID format. Expected EIN-XXXX-XXXX-XXXX")

    def _validate_temperature_readings(self, readings: List[float]) -> None:
        """
        Validate temperature sensor data
        
        Args:
            readings: List of temperature values in Celsius
            
        Raises:
            ValueError for unrealistic values
        """
        for temp in readings:
            if not (-50 <= temp <= 100):
                raise ValueError(f"Unrealistic temperature reading: {temp}°C")

    # ========================
    # Data Formatting Utilities
    # ========================
    
    def _format_shipment_for_web(self, raw_data: dict) -> dict:
        """
        Convert blockchain data to API-friendly format
        
        Args:
            raw_data: Raw data from blockchain
            
        Returns:
            Formatted dictionary with nested structure
        """
        return {
            "id": raw_data[0],
            "status": ShipmentStatus(raw_data[1]).name,
            "current_location": {
                "lat": raw_data[2][0],
                "lng": raw_data[2][1]
            },
            "timestamps": {
                "created": datetime.fromtimestamp(raw_data[3]),
                "updated": datetime.fromtimestamp(raw_data[4])
            }
        }

    def _parse_blockchain_timestamp(self, timestamp: int) -> datetime:
        """
        Convert blockchain timestamp to datetime
        
        Args:
            timestamp: Unix timestamp from blockchain
            
        Returns:
            Timezone-aware datetime object
        """
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def _calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate great-circle distance between two points (km)
        
        Args:
            point1: (lat, lng) tuple
            point2: (lat, lng) tuple
            
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = radians(point1[0]), radians(point1[1])
        lat2, lon2 = radians(point2[0]), radians(point2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return 6371 * c  # Earth radius in km
