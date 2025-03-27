package main

import (
	"encoding/json"
	"fmt"
	"time"
	"strings"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"github.com/hyperledger/fabric/common/flogging"
)

var logger = flogging.MustGetLogger("supplychain_cc")

// SmartContract extends the default contract with supply chain operations
type SmartContract struct {
	contractapi.Contract
}

// Asset Types
type Shipment struct {
	ID               string            `json:"id"`
	Items            []ShipmentItem    `json:"items"`
	Status           string            `json:"status"`
	CurrentLocation  GeoLocation       `json:"currentLocation"`
	Route            []RoutePoint      `json:"route"`
	TemperatureLogs  []TemperatureRead `json:"temperatureLogs"`
	Carrier          string            `json:"carrier"`
	InsuranceDetails Insurance         `json:"insurance"`
	Timestamps       EventTimestamps   `json:"timestamps"`
	Documents        []Document        `json:"documents"`
}

type ShipmentItem struct {
	SKU             string  `json:"sku"`
	Quantity        int     `json:"quantity"`
	BatchNumber     string  `json:"batchNumber"`
	ExpirationDate  string  `json:"expirationDate"`
	UnitPrice       float64 `json:"unitPrice"`
}

type GeoLocation struct {
	Latitude  float64 `json:"lat"`
	Longitude float64 `json:"lng"`
	Timestamp int64   `json:"timestamp"`
}

type RoutePoint struct {
	Location    GeoLocation `json:"location"`
	PlannedETA int64       `json:"eta"`
	ActualETA   int64       `json:"actualEta"`
}

type TemperatureRead struct {
	Timestamp int64   `json:"timestamp"`
	Value     float64 `json:"value"`
	SensorID  string  `json:"sensorId"`
}

type Insurance struct {
	Provider     string  `json:"provider"`
	PolicyNumber string  `json:"policyNumber"`
	Coverage     float64 `json:"coverage"`
}

type EventTimestamps struct {
	Created       int64 `json:"created"`
	Dispatched    int64 `json:"dispatched"`
	Delivered     int64 `json:"delivered"`
	CustomsCleared int64 `json:"customsCleared"`
}

type Document struct {
	Type        string `json:"type"` // invoice, billoflading, certificate
	Hash        string `json:"hash"`
	StorageURL  string `json:"storageUrl"`
	IssuedBy    string `json:"issuedBy"`
	IssuedDate  int64  `json:"issuedDate"`
}

// ========== Core Shipment Functions ==========

// CreateShipment initializes a new shipment with basic details
func (s *SmartContract) CreateShipment(ctx contractapi.TransactionContextInterface, 
	id string, itemsJSON string, carrier string, insuranceJSON string) error {
	
	caller, _ := ctx.GetClientIdentity().GetID()
	
	var items []ShipmentItem
	if err := json.Unmarshal([]byte(itemsJSON), &items); err != nil {
		return fmt.Errorf("failed to parse items: %v", err)
	}

	var insurance Insurance
	if err := json.Unmarshal([]byte(insuranceJSON), &insurance); err != nil {
		return fmt.Errorf("failed to parse insurance: %v", err)
	}

	shipment := Shipment{
		ID:      id,
		Items:   items,
		Status:  "CREATED",
		Carrier: carrier,
		InsuranceDetails: insurance,
		Timestamps: EventTimestamps{
			Created: time.Now().Unix(),
		},
	}

	if err := s.validateShipment(shipment); err != nil {
		return err
	}

	shipmentJSON, err := json.Marshal(shipment)
	if err != nil {
		return err
	}

	// Emit event for new shipment creation
	ctx.GetStub().SetEvent("ShipmentCreated", shipmentJSON)

	return ctx.GetStub().PutState(id, shipmentJSON)
}

// UpdateShipmentStatus updates the status with state validation
func (s *SmartContract) UpdateShipmentStatus(ctx contractapi.TransactionContextInterface, 
	id string, newStatus string) error {
	
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	// Validate state transitions
	validTransitions := map[string][]string{
		"CREATED":       {"DISPATCHED", "CANCELLED"},
		"DISPATCHED":    {"IN_TRANSIT", "DELAYED"},
		"IN_TRANSIT":    {"CUSTOMS_HOLD", "DELIVERED"},
		"CUSTOMS_HOLD":  {"RELEASED", "CONFISCATED"},
	}

	if !s.isValidTransition(shipment.Status, newStatus, validTransitions) {
		return fmt.Errorf("invalid status transition from %s to %s", shipment.Status, newStatus)
	}

	shipment.Status = newStatus
	s.updateTimestamps(shipment, newStatus)

	shipmentJSON, err := json.Marshal(shipment)
	if err != nil {
		return err
	}

	ctx.GetStub().SetEvent("ShipmentStatusChanged", []byte(
		fmt.Sprintf(`{"id":"%s","newStatus":"%s"}`, id, newStatus)))

	return ctx.GetStub().PutState(id, shipmentJSON)
}

// ========== Logistics Tracking ==========

// RecordLocation updates the current GPS position
func (s *SmartContract) RecordLocation(ctx contractapi.TransactionContextInterface, 
	id string, lat float64, lng float64) error {

	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	shipment.CurrentLocation = GeoLocation{
		Latitude:  lat,
		Longitude: lng,
		Timestamp: time.Now().Unix(),
	}

	return s.saveShipment(ctx, shipment)
}

// AddTemperatureReading stores IoT sensor data
func (s *SmartContract) AddTemperatureReading(ctx contractapi.TransactionContextInterface, 
	id string, value float64, sensorID string) error {

	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	reading := TemperatureRead{
		Timestamp: time.Now().Unix(),
		Value:     value,
		SensorID:  sensorID,
	}

	shipment.TemperatureLogs = append(shipment.TemperatureLogs, reading)

	// Check temperature violations
	if value < shipment.Items[0].MinimumTemp || value > shipment.Items[0].MaximumTemp {
		ctx.GetStub().SetEvent("TemperatureViolation", []byte(
			fmt.Sprintf(`{"id":"%s","value":%.2f}`, id, value)))
	}

	return s.saveShipment(ctx, shipment)
}

// ========== Document Management ==========

// AttachDocument links external documents to shipment
func (s *SmartContract) AttachDocument(ctx contractapi.TransactionContextInterface, 
	id string, docType string, hash string, url string) error {

	caller, _ := ctx.GetClientIdentity().GetID()
	
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	document := Document{
		Type:       docType,
		Hash:       hash,
		StorageURL: url,
		IssuedBy:   caller,
		IssuedDate: time.Now().Unix(),
	}

	shipment.Documents = append(shipment.Documents, document)

	return s.saveShipment(ctx, shipment)
}

// ========== Query Functions ==========

// GetShipmentHistory returns audit trail for a shipment
func (s *SmartContract) GetShipmentHistory(ctx contractapi.TransactionContextInterface, 
	id string) ([]HistoryEntry, error) {

	resultsIterator, err := ctx.GetStub().GetHistoryForKey(id)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var history []HistoryEntry
	for resultsIterator.HasNext() {
		response, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var entry HistoryEntry
		entry.TxID = response.TxId
		entry.Timestamp = response.Timestamp
		entry.IsDelete = response.IsDelete
		
		if !response.IsDelete {
			err = json.Unmarshal(response.Value, &entry.Shipment)
			if err != nil {
				return nil, err
			}
		}

		history = append(history, entry)
	}

	return history, nil
}

// QueryShipmentsByStatus returns all shipments matching a status
func (s *SmartContract) QueryShipmentsByStatus(ctx contractapi.TransactionContextInterface, 
	status string) ([]Shipment, error) {

	query := fmt.Sprintf(`{"selector":{"status":"%s"}}`, status)
	return s.richQuery(ctx, query)
}

// ========== Helper Functions ==========

func (s *SmartContract) GetShipment(ctx contractapi.TransactionContextInterface, 
	id string) (*Shipment, error) {

	shipmentJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if shipmentJSON == nil {
		return nil, fmt.Errorf("shipment %s does not exist", id)
	}

	var shipment Shipment
	err = json.Unmarshal(shipmentJSON, &shipment)
	if err != nil {
		return nil, err
	}

	return &shipment, nil
}

func (s *SmartContract) saveShipment(ctx contractapi.TransactionContextInterface, 
	shipment *Shipment) error {

	shipmentJSON, err := json.Marshal(shipment)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(shipment.ID, shipmentJSON)
}

// ========== Main Function ==========

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		logger.Panicf("Error creating supply chain chaincode: %v", err)
	}

	if err := chaincode.Start(); err != nil {
		logger.Panicf("Error starting supply chain chaincode: %v", err)
	}
}

// ========== Validation & Security ==========

const (
	RoleCarrier   = "CARRIER"
	RoleSupplier  = "SUPPLIER"
	RoleRegulator = "REGULATOR"
)

func (s *SmartContract) validateShipment(shipment Shipment) error {
	if shipment.ID == "" {
		return fmt.Errorf("shipment ID cannot be empty")
	}
	
	if len(shipment.Items) == 0 {
		return fmt.Errorf("shipment must contain at least one item")
	}

	for _, item := range shipment.Items {
		if item.Quantity <= 0 {
			return fmt.Errorf("invalid quantity for item %s", item.SKU)
		}
	}

	if shipment.InsuranceDetails.Coverage < 0 {
		return fmt.Errorf("insurance coverage cannot be negative")
	}

	return nil
}

func (s *SmartContract) isValidTransition(oldStatus, newStatus string, transitions map[string][]string) bool {
	allowed, exists := transitions[oldStatus]
	if !exists {
		return false
	}

	for _, status := range allowed {
		if status == newStatus {
			return true
		}
	}
	return false
}

func (s *SmartContract) checkRole(ctx contractapi.TransactionContextInterface, requiredRole string) error {
	attrs, err := ctx.GetClientIdentity().GetAttributes()
	if err != nil {
		return err
	}

	for _, attr := range attrs {
		if attr.Name == "role" && attr.Value == requiredRole {
			return nil
		}
	}
	
	return fmt.Errorf("caller doesn't have required role: %s", requiredRole)
}

// ========== Route Management ==========

func (s *SmartContract) AddRoutePoint(ctx contractapi.TransactionContextInterface, 
	id string, lat float64, lng float64, eta int64) error {

	if err := s.checkRole(ctx, RoleCarrier); err != nil {
		return err
	}

	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	newPoint := RoutePoint{
		Location: GeoLocation{
			Latitude:  lat,
			Longitude: lng,
			Timestamp: time.Now().Unix(),
		},
		PlannedETA: eta,
	}

	shipment.Route = append(shipment.Route, newPoint)
	return s.saveShipment(ctx, shipment)
}

func (s *SmartContract) CalculateETA(ctx contractapi.TransactionContextInterface, id string) (int64, error) {
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return 0, err
	}

	if len(shipment.Route) == 0 {
		return 0, fmt.Errorf("no route points available")
	}

	// Simplified ETA calculation (real implementation would use distance matrix API)
	lastPoint := shipment.Route[len(shipment.Route)-1]
	return lastPoint.PlannedETA, nil
}

// ========== Compliance & Safety ==========

func (s *SmartContract) CheckCustomsCompliance(ctx contractapi.TransactionContextInterface, id string) (bool, error) {
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return false, err
	}

	requiredDocs := map[string]bool{
		"commercial_invoice": false,
		"certificate_of_origin": false,
	}

	for _, doc := range shipment.Documents {
		if _, exists := requiredDocs[doc.Type]; exists {
			requiredDocs[doc.Type] = true
		}
	}

	for docType, present := range requiredDocs {
		if !present {
			return false, fmt.Errorf("missing required document: %s", docType)
		}
	}

	return true, nil
}

func (s *SmartContract) CheckTemperatureCompliance(ctx contractapi.TransactionContextInterface, id string) error {
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	for _, reading := range shipment.TemperatureLogs {
		for _, item := range shipment.Items {
			if reading.Value < item.MinTemp || reading.Value > item.MaxTemp {
				return fmt.Errorf("temperature violation detected: %.2f°C", reading.Value)
			}
		}
	}

	return nil
}

// ========== Advanced Features ==========

func (s *SmartContract) FileInsuranceClaim(ctx contractapi.TransactionContextInterface, 
	id string, claimAmount float64, description string) error {

	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	if claimAmount > shipment.InsuranceDetails.Coverage {
		return fmt.Errorf("claim amount exceeds coverage limit")
	}

	claim := InsuranceClaim{
		ID:          fmt.Sprintf("CLM-%d", time.Now().Unix()),
		ShipmentID:  id,
		Amount:      claimAmount,
		Description: description,
		Status:      "PENDING",
		FiledAt:     time.Now().Unix(),
	}

	claimJSON, _ := json.Marshal(claim)
	ctx.GetStub().SetEvent("InsuranceClaimFiled", claimJSON)

	return s.saveInsuranceClaim(ctx, claim)
}

func (s *SmartContract) AddBatchInformation(ctx contractapi.TransactionContextInterface, 
	id string, sku string, batch BatchInfo) error {

	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	for i, item := range shipment.Items {
		if item.SKU == sku {
			shipment.Items[i].BatchInfo = batch
			return s.saveShipment(ctx, shipment)
		}
	}

	return fmt.Errorf("SKU %s not found in shipment", sku)
}

// ========== Query Enhancements ==========

func (s *SmartContract) QueryByGeoRange(ctx contractapi.TransactionContextInterface, 
	lat float64, lng float64, distanceKm int) ([]Shipment, error) {

	query := fmt.Sprintf(`{
		"selector": {
			"currentLocation": {
				"$geoWithin": {
					"$center": [[%f, %f], %d]
				}
			}
		}
	}`, lat, lng, distanceKm)

	return s.richQuery(ctx, query)
}

func (s *SmartContract) QueryExpiringShipments(ctx contractapi.TransactionContextInterface, 
	days int) ([]Shipment, error) {

	expirationThreshold := time.Now().AddDate(0, 0, days).Unix()
	query := fmt.Sprintf(`{
		"selector": {
			"items": {
				"$elemMatch": {
					"expirationDate": {"$lt": %d}
				}
			}
		}
	}`, expirationThreshold)

	return s.richQuery(ctx, query)
}

// ========== Utility Functions ==========

func (s *SmartContract) richQuery(ctx contractapi.TransactionContextInterface, query string) ([]Shipment, error) {
	resultsIterator, err := ctx.GetStub().GetQueryResult(query)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var shipments []Shipment
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var shipment Shipment
		err = json.Unmarshal(queryResponse.Value, &shipment)
		if err != nil {
			return nil, err
		}
		shipments = append(shipments, shipment)
	}

	return shipments, nil
}

func (s *SmartContract) updateTimestamps(shipment *Shipment, newStatus string) {
	now := time.Now().Unix()
	switch newStatus {
	case "DISPATCHED":
		shipment.Timestamps.Dispatched = now
	case "DELIVERED":
		shipment.Timestamps.Delivered = now
	case "CUSTOMS_CLEARED":
		shipment.Timestamps.CustomsCleared = now
	}
}

// ========== Additional Structs ==========

type InsuranceClaim struct {
	ID          string  `json:"id"`
	ShipmentID  string  `json:"shipmentId"`
	Amount      float64 `json:"amount"`
	Description string  `json:"description"`
	Status      string  `json:"status"`
	FiledAt     int64   `json:"filedAt"`
	ResolvedAt  int64   `json:"resolvedAt"`
}

type BatchInfo struct {
	Manufacturer    string `json:"manufacturer"`
	ProductionDate  string `json:"productionDate"`
	ExpirationDate  string `json:"expirationDate"`
	QCStatus        string `json:"qcStatus"`
}

type HistoryEntry struct {
	TxID      string    `json:"txId"`
	Timestamp time.Time `json:"timestamp"`
	Shipment  Shipment  `json:"shipment"`
	IsDelete  bool      `json:"isDelete"`
}

// ========== Index Definitions ==========

func (s *SmartContract) CreateIndexes(ctx contractapi.TransactionContextInterface) error {
	indexes := []string{
		`{
			"index": {
				"fields": ["status"]
			},
			"name": "statusIndex",
			"type": "json"
		}`,
		`{
			"index": {
				"fields": ["currentLocation"],
				"geo": true
			},
			"name": "geoIndex",
			"type": "json"
		}`,
	}

	for _, index := range indexes {
		err := ctx.GetStub().CreateIndex("shipmentIndex", index)
		if err != nil {
			return err
		}
	}

	return nil
}

// ========== Maintenance Operations ==========

func (s *SmartContract) ArchiveShipment(ctx contractapi.TransactionContextInterface, id string) error {
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	if shipment.Status != "DELIVERED" {
		return fmt.Errorf("cannot archive undelivered shipment")
	}

	shipmentJSON, _ := json.Marshal(shipment)
	ctx.GetStub().SetEvent("ShipmentArchived", shipmentJSON)

	return ctx.GetStub().DelState(id)
}

// ========== Multi-Party Authorization ==========

const (
	RequiredCustomsApprovals = 2
)

func (s *SmartContract) ApproveCustomsClearance(ctx contractapi.TransactionContextInterface, 
	id string, agency string) error {

	// Validate regulator role
	if err := s.checkRole(ctx, RoleRegulator); err != nil {
		return err
	}

	// Get current shipment state
	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return err
	}

	// Check for existing approval from this agency
	for _, approval := range shipment.CustomsApprovals {
		if approval.Agency == agency {
			return fmt.Errorf("customs approval already granted by %s", agency)
		}
	}

	// Add new approval
	shipment.CustomsApprovals = append(shipment.CustomsApprovals, CustomsApproval{
		Agency:     agency,
		ApprovedAt: time.Now().Unix(),
		Signature:  ctx.GetClientIdentity().GetMSPID(),
	})

	// Update customs status if requirements met
	if len(shipment.CustomsApprovals) >= RequiredCustomsApprovals {
		shipment.Status = "CUSTOMS_CLEARED"
		ctx.GetStub().SetEvent("CustomsCleared", []byte(id))
	}

	return s.saveShipment(ctx, shipment)
}

type CustomsApproval struct {
	Agency     string `json:"agency"`
	ApprovedAt int64  `json:"approvedAt"`
	Signature  string `json:"signature"` // MSP ID of approving organization
}

// ========== Carbon Footprint Tracking ==========

const (
	EarthRadiusKm       = 6371.0
	TruckEmissionFactor = 0.206 // kg CO2/km per ton of cargo
	ShipEmissionFactor  = 0.010 // kg CO2/km per ton of cargo
	AirEmissionFactor   = 0.500 // kg CO2/km per ton of cargo
)

func (s *SmartContract) CalculateCarbonFootprint(ctx contractapi.TransactionContextInterface, 
	id string) (float64, error) {

	shipment, err := s.GetShipment(ctx, id)
	if err != nil {
		return 0, err
	}

	// Calculate total distance in kilometers
	totalDistance, err := calculateRouteDistance(shipment.Route)
	if err != nil {
		return 0, fmt.Errorf("distance calculation failed: %v", err)
	}

	// Calculate total cargo weight
	totalWeight := 0.0
	for _, item := range shipment.Items {
		totalWeight += item.Weight * float64(item.Quantity)
	}

	// Select emission factor based on transport mode
	emissionFactor, err := getEmissionFactor(shipment.TransportMode)
	if err != nil {
		return 0, err
	}

	// Calculate CO2 emissions: distance(km) × weight(t) × factor(kg CO2/km/t)
	return totalDistance * (totalWeight / 1000) * emissionFactor, nil
}

func calculateRouteDistance(route []RoutePoint) (float64, error) {
	if len(route) < 2 {
		return 0.0, nil
	}

	totalDistance := 0.0
	for i := 1; i < len(route); i++ {
		prev := route[i-1].Location
		current := route[i].Location

		// Convert degrees to radians
		lat1 := degreesToRadians(prev.Latitude)
		lon1 := degreesToRadians(prev.Longitude)
		lat2 := degreesToRadians(current.Latitude)
		lon2 := degreesToRadians(current.Longitude)

		// Haversine formula
		dlat := lat2 - lat1
		dlon := lon2 - lon1

		a := math.Pow(math.Sin(dlat/2), 2) + math.Cos(lat1)*math.Cos(lat2)*
			math.Pow(math.Sin(dlon/2), 2)
		c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

		distance := EarthRadiusKm * c
		totalDistance += distance
	}

	return totalDistance, nil
}

func degreesToRadians(deg float64) float64 {
	return deg * math.Pi / 180
}

func getEmissionFactor(transportMode string) (float64, error) {
	switch strings.ToUpper(transportMode) {
	case "TRUCK":
		return TruckEmissionFactor, nil
	case "SHIP":
		return ShipEmissionFactor, nil
	case "AIR":
		return AirEmissionFactor, nil
	default:
		return 0, fmt.Errorf("unknown transport mode: %s", transportMode)
	}
}

// Add to Shipment struct:
type Shipment struct {
	// ... existing fields ...
	TransportMode    string           `json:"transportMode"`
	CustomsApprovals []CustomsApproval `json:"customsApprovals"`
}

// Add to ShipmentItem struct:
type ShipmentItem struct {
	// ... existing fields ...
	Weight      float64 `json:"weight"` // in kilograms
}
