// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

interface ISupplyChain {
    event ShipmentCreated(string indexed shipmentId, address indexed supplier);
    event ShipmentUpdated(string indexed shipmentId, string status, string location, uint256 timestamp);
    event ShipmentDelivered(string indexed shipmentId, uint256 deliveryTimestamp);
    event RoleGranted(bytes32 indexed role, address indexed account);
    event ShipmentDeleted(string indexed shipmentId);
    event EmergencyStopActivated(address indexed admin);
    event EmergencyStopLifted(address indexed admin);
    
    error UnauthorizedAccess(address caller);
    error ShipmentAlreadyExists(string shipmentId);
    error ShipmentNotFound(string shipmentId);
    error InvalidParameters();
    error ContractPaused();
}

contract SupplyChain is ISupplyChain, AccessControl {
    using Counters for Counters.Counter;
    
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant SUPPLIER_ROLE = keccak256("SUPPLIER_ROLE");
    bytes32 public constant CARRIER_ROLE = keccak256("CARRIER_ROLE");
    
    struct Shipment {
        address supplier;
        address carrier;
        uint256 creationTimestamp;
        uint256 lastUpdateTimestamp;
        uint256 estimatedTimeOfArrival;
        string currentStatus;
        string currentLocation;
        string origin;
        string destination;
        string[] itemCodes;
        bool isHazardous;
        bool requiresCustoms;
        bool isTemperatureControlled;
    }
    
    struct ShipmentHistory {
        string status;
        string location;
        uint256 timestamp;
        address updatedBy;
    }
    
    bool public contractPaused;
    Counters.Counter private _totalShipments;
    mapping(string => Shipment) public shipments;
    mapping(string => ShipmentHistory[]) private _shipmentHistory;
    mapping(address => string[]) private _supplierShipments;
    mapping(address => string[]) private _carrierShipments;
    string[] private _allShipmentIds;
    
    modifier onlyAdmin() {
        if (!hasRole(ADMIN_ROLE, msg.sender)) revert UnauthorizedAccess(msg.sender);
        _;
    }
    
    modifier onlyActive() {
        if (contractPaused) revert ContractPaused();
        _;
    }
    
    modifier validAddress(address addr) {
        if (addr == address(0)) revert InvalidParameters();
        _;
    }
    
    constructor() {
        _setupRole(ADMIN_ROLE, msg.sender);
        _setRoleAdmin(ADMIN_ROLE, ADMIN_ROLE);
        _setRoleAdmin(SUPPLIER_ROLE, ADMIN_ROLE);
        _setRoleAdmin(CARRIER_ROLE, ADMIN_ROLE);
    }
    
    function createShipment(
        string calldata shipmentId,
        string calldata origin,
        string calldata destination,
        uint256 estimatedTimeOfArrival,
        string[] calldata itemCodes,
        bool isHazardous,
        bool requiresCustoms,
        bool isTemperatureControlled
    ) external onlyActive onlyRole(SUPPLIER_ROLE) {
        if (bytes(shipmentId).length == 0) revert InvalidParameters();
        if (shipments[shipmentId].supplier != address(0)) revert ShipmentAlreadyExists(shipmentId);
        
        Shipment storage newShipment = shipments[shipmentId];
        newShipment.supplier = msg.sender;
        newShipment.origin = origin;
        newShipment.destination = destination;
        newShipment.estimatedTimeOfArrival = estimatedTimeOfArrival;
        newShipment.itemCodes = itemCodes;
        newShipment.isHazardous = isHazardous;
        newShipment.requiresCustoms = requiresCustoms;
        newShipment.isTemperatureControlled = isTemperatureControlled;
        newShipment.creationTimestamp = block.timestamp;
        newShipment.lastUpdateTimestamp = block.timestamp;
        newShipment.currentStatus = "CREATED";
        newShipment.currentLocation = origin;
        
        _shipmentHistory[shipmentId].push(ShipmentHistory({
            status: "CREATED",
            location: origin,
            timestamp: block.timestamp,
            updatedBy: msg.sender
        }));
        
        _totalShipments.increment();
        _allShipmentIds.push(shipmentId);
        _supplierShipments[msg.sender].push(shipmentId);
        
        emit ShipmentCreated(shipmentId, msg.sender);
    }
    
    function updateShipmentStatus(
        string calldata shipmentId,
        string calldata newStatus,
        string calldata newLocation,
        uint256 newETA
    ) external onlyActive {
        Shipment storage shipment = _getShipment(shipmentId);
        
        if (msg.sender != shipment.supplier && 
            msg.sender != shipment.carrier && 
            !hasRole(ADMIN_ROLE, msg.sender)) {
            revert UnauthorizedAccess(msg.sender);
        }
        
        shipment.currentStatus = newStatus;
        shipment.currentLocation = newLocation;
        shipment.lastUpdateTimestamp = block.timestamp;
        shipment.estimatedTimeOfArrival = newETA;
        
        _shipmentHistory[shipmentId].push(ShipmentHistory({
            status: newStatus,
            location: newLocation,
            timestamp: block.timestamp,
            updatedBy: msg.sender
        }));
        
        if (keccak256(bytes(newStatus)) == keccak256(bytes("DELIVERED"))) {
            emit ShipmentDelivered(shipmentId, block.timestamp);
        }
        
        emit ShipmentUpdated(shipmentId, newStatus, newLocation, block.timestamp);
    }
    
    function assignCarrier(
        string calldata shipmentId, 
        address carrier
    ) external onlyActive validAddress(carrier) {
        Shipment storage shipment = _getShipment(shipmentId);
        if (msg.sender != shipment.supplier) revert UnauthorizedAccess(msg.sender);
        if (!hasRole(CARRIER_ROLE, carrier)) revert UnauthorizedAccess(carrier);
        
        shipment.carrier = carrier;
        _carrierShipments[carrier].push(shipmentId);
        
        _shipmentHistory[shipmentId].push(ShipmentHistory({
            status: "CARRIER_ASSIGNED",
            location: shipment.currentLocation,
            timestamp: block.timestamp,
            updatedBy: msg.sender
        }));
        
        emit ShipmentUpdated(shipmentId, "CARRIER_ASSIGNED", shipment.currentLocation, block.timestamp);
    }
    
    function getShipmentHistory(
        string calldata shipmentId
    ) external view returns (ShipmentHistory[] memory) {
        return _shipmentHistory[shipmentId];
    }
    
    function getAllShipments() external view returns (string[] memory) {
        return _allShipmentIds;
    }
    
    function getShipmentCount() external view returns (uint256) {
        return _totalShipments.current();
    }
    
    function grantRole(
        bytes32 role, 
        address account
    ) public override onlyAdmin validAddress(account) {
        super.grantRole(role, account);
        emit RoleGranted(role, account);
    }
    
    function emergencyStop(bool pause) external onlyAdmin {
        contractPaused = pause;
        if (pause) {
            emit EmergencyStopActivated(msg.sender);
        } else {
            emit EmergencyStopLifted(msg.sender);
        }
    }
    
    function deleteShipment(string calldata shipmentId) external onlyAdmin {
        if (shipments[shipmentId].supplier == address(0)) revert ShipmentNotFound(shipmentId);
        
        delete shipments[shipmentId];
        delete _shipmentHistory[shipmentId];
        
        for (uint256 i = 0; i < _allShipmentIds.length; i++) {
            if (keccak256(bytes(_allShipmentIds[i])) == keccak256(bytes(shipmentId))) {
                _allShipmentIds[i] = _allShipmentIds[_allShipmentIds.length - 1];
                _allShipmentIds.pop();
                break;
            }
        }
        
        _totalShipments.decrement();
        emit ShipmentDeleted(shipmentId);
    }
    
    function _getShipment(
        string calldata shipmentId
    ) private view returns (Shipment storage) {
        if (shipments[shipmentId].supplier == address(0)) revert ShipmentNotFound(shipmentId);
        return shipments[shipmentId];
    }
    
    function getSupplierShipments(
        address supplier
    ) external view validAddress(supplier) returns (string[] memory) {
        return _supplierShipments[supplier];
    }
    
    function getCarrierShipments(
        address carrier
    ) external view validAddress(carrier) returns (string[] memory) {
        return _carrierShipments[carrier];
    }
    
    function getShipmentDetails(
        string calldata shipmentId
    ) external view returns (
        address supplier,
        address carrier,
        uint256 creationTimestamp,
        uint256 lastUpdateTimestamp,
        string memory currentStatus,
        string memory currentLocation,
        string memory origin,
        string memory destination
    ) {
        Shipment storage shipment = _getShipment(shipmentId);
        return (
            shipment.supplier,
            shipment.carrier,
            shipment.creationTimestamp,
            shipment.lastUpdateTimestamp,
            shipment.currentStatus,
            shipment.currentLocation,
            shipment.origin,
            shipment.destination
        );
    }
    
    function getShipmentMetadata(
        string calldata shipmentId
    ) external view returns (
        bool isHazardous,
        bool requiresCustoms,
        bool isTemperatureControlled,
        string[] memory itemCodes
    ) {
        Shipment storage shipment = _getShipment(shipmentId);
        return (
            shipment.isHazardous,
            shipment.requiresCustoms,
            shipment.isTemperatureControlled,
            shipment.itemCodes
        );
    }
    
    function batchUpdateShipments(
        string[] calldata shipmentIds,
        string[] calldata newStatuses,
        string[] calldata newLocations
    ) external onlyActive onlyRole(ADMIN_ROLE) {
        if (shipmentIds.length != newStatuses.length || 
            shipmentIds.length != newLocations.length) revert InvalidParameters();
            
        for (uint256 i = 0; i < shipmentIds.length; i++) {
            Shipment storage shipment = shipments[shipmentIds[i]];
            if (shipment.supplier == address(0)) continue;
            
            shipment.currentStatus = newStatuses[i];
            shipment.currentLocation = newLocations[i];
            shipment.lastUpdateTimestamp = block.timestamp;
            
            _shipmentHistory[shipmentIds[i]].push(ShipmentHistory({
                status: newStatuses[i],
                location: newLocations[i],
                timestamp: block.timestamp,
                updatedBy: msg.sender
            }));
            
            emit ShipmentUpdated(shipmentIds[i], newStatuses[i], newLocations[i], block.timestamp);
        }
    }
    
    function destructContract() external onlyAdmin {
        selfdestruct(payable(msg.sender));
    }
}
