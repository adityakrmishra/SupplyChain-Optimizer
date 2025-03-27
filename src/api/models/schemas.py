"""
Supply Chain API Schemas
------------------------
Defines Pydantic models for:
- Shipment tracking
- Inventory management
- Customs compliance
- IoT sensor monitoring
- User authentication
- Financial transactions
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, condecimal, confloat, conint

# ====================
# Enumerated Types
# ====================

class ShipmentStatus(str, Enum):
    """Lifecycle states for shipments"""
    DRAFT = "DRAFT"
    AWAITING_PICKUP = "AWAITING_PICKUP"
    IN_TRANSIT = "IN_TRANSIT"
    CUSTOMS_HOLD = "CUSTOMS_HOLD"
    WAREHOUSE_HOLD = "WAREHOUSE_HOLD"
    DELAYED = "DELAYED"
    PARTIALLY_DELIVERED = "PARTIALLY_DELIVERED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

class HazardClass(str, Enum):
    """UN hazardous material classification"""
    CLASS_1 = "Explosives"
    CLASS_2 = "Gases"
    CLASS_3 = "Flammable Liquids"
    CLASS_4 = "Flammable Solids"
    CLASS_5 = "Oxidizing Substances"
    CLASS_6 = "Toxic Substances"
    CLASS_7 = "Radioactive Material"
    CLASS_8 = "Corrosives"
    CLASS_9 = "Miscellaneous"

class TransportMode(str, Enum):
    """Shipping transportation types"""
    AIR = "AIR"
    OCEAN = "OCEAN"
    TRUCK = "TRUCK"
    RAIL = "RAIL"
    INTERMODAL = "INTERMODAL"

class CurrencyCode(str, Enum):
    """ISO 4217 currency codes"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"

class DocumentType(str, Enum):
    """Supply chain document types"""
    BILL_OF_LADING = "BILL_OF_LADING"
    COMMERCIAL_INVOICE = "COMMERCIAL_INVOICE"
    PACKING_LIST = "PACKING_LIST"
    CERTIFICATE_OF_ORIGIN = "CERTIFICATE_OF_ORIGIN"
    INSURANCE_CERTIFICATE = "INSURANCE_CERTIFICATE"
    CUSTOMS_DECLARATION = "CUSTOMS_DECLARATION"

class SensorType(str, Enum):
    """IoT sensor measurement types"""
    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    SHOCK = "SHOCK"
    LIGHT_EXPOSURE = "LIGHT_EXPOSURE"
    TILT = "TILT"
    LOCATION = "LOCATION"

# ====================
# Core Shipment Models
# ====================

class ContactInfo(BaseModel):
    """Contact information for supply chain parties"""
    company: str = Field(..., max_length=100)
    contact_name: str = Field(..., max_length=50)
    email: str = Field(..., regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    phone: str = Field(..., regex=r"^\+?[1-9]\d{1,14}$")
    address: str = Field(..., max_length=200)
    city: str = Field(..., max_length=50)
    country: str = Field(..., max_length=50)

class GeoLocation(BaseModel):
    """Geospatial coordinates with validation"""
    lat: confloat(ge=-90, le=90) = Field(..., example=37.7749)
    lng: confloat(ge=-180, le=180) = Field(..., example=-122.4194)
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('lat', 'lng')
    def round_coordinates(cls, v):
        """Round coordinates to 6 decimal places (~10cm precision)"""
        return round(v, 6)

class ShipmentItem(BaseModel):
    """Individual line items in a shipment"""
    sku: str = Field(..., min_length=4, max_length=20)
    description: str = Field(..., max_length=200)
    quantity: conint(gt=0) = Field(..., example=100)
    unit_weight_kg: condecimal(gt=0, decimal_places=3) = Field(..., example=0.5)
    unit_value: condecimal(ge=0, decimal_places=2) = Field(..., example=10.99)
    hazardous: bool = False
    hazard_class: Optional[HazardClass]
    temperature_range: Optional[tuple[confloat(), confloat()]]
    
    @validator('temperature_range')
    def validate_temperature_range(cls, v):
        if v and v[0] >= v[1]:
            raise ValueError("Minimum temperature must be less than maximum")
        return v

class TransportationSegment(BaseModel):
    """Individual legs of a multi-modal shipment"""
    sequence: conint(gt=0)
    mode: TransportMode
    carrier: str
    vessel_flight_number: Optional[str]
    departure_location: GeoLocation
    arrival_location: GeoLocation
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: Optional[datetime]
    actual_arrival: Optional[datetime]

class ShipmentDocument(BaseModel):
    """Digital documents associated with shipments"""
    type: DocumentType
    document_id: str = Field(..., min_length=10)
    issue_date: datetime
    expiration_date: Optional[datetime]
    issuing_authority: str
    storage_url: str
    hash_sha256: str = Field(..., regex=r"^[a-f0-9]{64}$")

class ShipmentBase(BaseModel):
    """Base shipment model with common fields"""
    origin: ContactInfo
    destination: ContactInfo
    items: List[ShipmentItem] = Field(..., min_items=1)
    transport_mode: TransportMode
    incoterm: str = Field(..., example="FOB", max_length=3)
    required_documents: List[DocumentType]
    special_instructions: Optional[str]

# ====================
# Shipment Sub-models
# ====================

class ShipmentCosts(BaseModel):
    """Breakdown of financial components"""
    shipping_cost: condecimal(ge=0, decimal_places=2)
    insurance_cost: condecimal(ge=0, decimal_places=2)
    customs_duties: condecimal(ge=0, decimal_places=2)
    taxes: condecimal(ge=0, decimal_places=2)
    currency: CurrencyCode = CurrencyCode.USD

    @property
    def total_cost(self):
        return sum([self.shipping_cost, self.insurance_cost, 
                   self.customs_duties, self.taxes])

class ShipmentInsurance(BaseModel):
    """Insurance policy details"""
    policy_number: str
    provider: str
    coverage_amount: condecimal(ge=0)
    coverage_type: str
    deductible: condecimal(ge=0)
    certificate_url: str

class CustomsInfo(BaseModel):
    """Customs clearance information"""
    hs_code: str = Field(..., regex=r"^\d{6,10}$")
    export_control_class: Optional[str]
    import_country: str
    export_country: str
    customs_value: condecimal(ge=0)
    duties_paid: bool = False
    clearance_status: Optional[str]

class SensorReading(BaseModel):
    """IoT device sensor measurements"""
    sensor_id: str
    type: SensorType
    value: confloat()
    timestamp: datetime
    unit: str
    coordinates: Optional[GeoLocation]

    @validator('unit')
    def validate_unit(cls, v, values):
        unit_map = {
            SensorType.TEMPERATURE: "°C",
            SensorType.HUMIDITY: "%",
            SensorType.SHOCK: "g",
            SensorType.LIGHT_EXPOSURE: "lux",
            SensorType.TILT: "degrees"
        }
        if values.get('type') and v != unit_map.get(values['type']):
            raise ValueError(f"Invalid unit {v} for {values['type']}")
        return v

# ====================
# Main Shipment Model
# ====================

class ShipmentCreate(ShipmentBase):
    """Shipment creation schema"""
    scheduled_departure: datetime
    estimated_arrival: datetime
    transportation_plan: List[TransportationSegment]
    insurance: Optional[ShipmentInsurance]
    customs_info: CustomsInfo
    costs: ShipmentCosts

    @validator('estimated_arrival')
    def validate_arrival_date(cls, v, values):
        if 'scheduled_departure' in values and v <= values['scheduled_departure']:
            raise ValueError("Arrival date must be after departure")
        return v

class ShipmentUpdate(BaseModel):
    """Shipment update fields"""
    current_location: Optional[GeoLocation]
    status: Optional[ShipmentStatus]
    delay_reason: Optional[str]
    actual_arrival: Optional[datetime]
    updated_documents: Optional[List[ShipmentDocument]]
    sensor_readings: Optional[List[SensorReading]]

class ShipmentResponse(ShipmentBase):
    """Full shipment response with system-generated fields"""
    id: str = Field(..., example="SC-123456")
    tracking_number: str
    status: ShipmentStatus
    created_at: datetime
    updated_at: datetime
    transportation_history: List[TransportationSegment]
    documents: List[ShipmentDocument]
    sensor_data: List[SensorReading]
    costs: ShipmentCosts
    insurance: Optional[ShipmentInsurance]
    customs_status: CustomsInfo
    events: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "SC-123456",
                "tracking_number": "TN-987654321",
                "status": "IN_TRANSIT",
                "created_at": "2023-07-20T09:00:00Z",
                "updated_at": "2023-07-21T14:30:00Z"
            }
        }

# ====================
# User Authentication
# ====================

class UserRoles(str, Enum):
    """System user roles"""
    ADMIN = "ADMIN"
    LOGISTICS_MANAGER = "LOGISTICS_MANAGER"
    CUSTOMS_AGENT = "CUSTOMS_AGENT"
    CARRIER = "CARRIER"
    VIEWER = "VIEWER"

class UserBase(BaseModel):
    """Base user model"""
    email: str = Field(..., regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    company: str = Field(..., max_length=100)
    phone: Optional[str]
    timezone: str = "UTC"

class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=12)
    role: UserRoles = UserRoles.VIEWER
    two_factor_enabled: bool = False

class UserResponse(UserBase):
    """User response model"""
    id: int
    role: UserRoles
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    permissions: List[str]

    class Config:
        orm_mode = True

# ====================
# Security Models
# ====================

class Token(BaseModel):
    """Authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str]

class TokenPayload(BaseModel):
    """JWT token payload structure"""
    sub: Optional[int] = None
    exp: Optional[int] = None
    scopes: List[str] = []

class PasswordResetRequest(BaseModel):
    """Password reset initiation"""
    email: str

class PasswordResetConfirm(BaseModel):
    """Password reset completion"""
    token: str
    new_password: str = Field(..., min_length=12)

# ====================
# System Events
# ====================

class SystemEventType(str, Enum):
    """Audit log event types"""
    USER_LOGIN = "USER_LOGIN"
    SHIPMENT_UPDATE = "SHIPMENT_UPDATE"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    ALERT_TRIGGERED = "ALERT_TRIGGERED"
    API_CALL = "API_CALL"

class AuditLogEntry(BaseModel):
    """Security audit log record"""
    timestamp: datetime
    event_type: SystemEventType
    user_id: Optional[int]
    description: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    metadata: Dict[str, Any]

# ====================
# Notification System
# ====================

class NotificationType(str, Enum):
    """Alert notification types"""
    DELAY_ALERT = "DELAY_ALERT"
    CUSTOMS_HOLD = "CUSTOMS_HOLD"
    TEMPERATURE_BREACH = "TEMPERATURE_BREACH"
    DOCUMENT_EXPIRATION = "DOCUMENT_EXPIRATION"

class Notification(BaseModel):
    """System-generated alert"""
    type: NotificationType
    shipment_id: str
    message: str
    severity: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[int]
    metadata: Dict[str, Any]

# ====================
# API Operations
# ====================

class Pagination(BaseModel):
    """API pagination controls"""
    page: conint(ge=1) = 1
    page_size: conint(ge=10, le=100) = 50

class ShipmentFilter(BaseModel):
    """Shipment search filters"""
    status: Optional[ShipmentStatus]
    origin_country: Optional[str]
    destination_country: Optional[str]
    created_after: Optional[datetime]
    created_before: Optional[datetime]
    carrier: Optional[str]

class BulkOperationResponse(BaseModel):
    """Bulk operation result"""
    success_count: int
    error_count: int
    errors: List[Dict[str, str]]

# ====================
# Financial Models
# ====================

class Invoice(BaseModel):
    """Commercial invoice details"""
    invoice_number: str
    issue_date: datetime
    due_date: datetime
    total_amount: condecimal(ge=0)
    currency: CurrencyCode
    payment_status: str
    payment_terms: str

class PaymentRecord(BaseModel):
    """Payment transaction record"""
    transaction_id: str
    amount: condecimal(ge=0)
    currency: CurrencyCode
    payment_date: datetime
    method: str
    reference_number: str

# ====================
# Compliance Models
# ====================

class RegulatoryCheck(BaseModel):
    """Compliance verification record"""
    check_type: str
    passed: bool
    checked_by: str
    checked_at: datetime
    valid_until: datetime
    certificate_url: Optional[str]

class SanctionScreeningResult(BaseModel):
    """Trade sanction check results"""
    screened_parties: List[str]
    match_found: bool
    screened_at: datetime
    reference_id: str

# ====================
# Equipment Models
# ====================

class ContainerSpec(BaseModel):
    """Shipping container specifications"""
    container_type: str
    iso_code: str
    tare_weight: condecimal(ge=0)
    max_payload: condecimal(ge=0)
    temperature_capable: bool
    last_inspection: datetime

class VehicleInfo(BaseModel):
    """Transport vehicle details"""
    vehicle_type: str
    identification_number: str
    carrier: str
    current_location: Optional[GeoLocation]
    last_maintenance: datetime

# ====================
# Custom Validators
# ====================

def validate_hazardous_materials(items: List[ShipmentItem]):
    """Validate hazardous material declarations"""
    for item in items:
        if item.hazardous and not item.hazard_class:
            raise ValueError("Hazard class required for dangerous goods")
    return items

def validate_incoterm(incoterm: str):
    """Validate INCOTERM format"""
    valid_incoterms = ["EXW", "FCA", "FAS", "FOB", "CFR", "CIF", 
                      "CPT", "CIP", "DAT", "DAP", "DDP"]
    if incoterm not in valid_incoterms:
        raise ValueError("Invalid INCOTERM")
    return incoterm

# ====================
# Response Wrappers
# ====================

class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    warnings: List[str]

class HealthCheckResponse(BaseModel):
    """System health status"""
    status: str
    version: str
    database_status: str
    cache_status: str
    uptime: float

# ====================
# Configuration Models
# ====================

class SystemConfiguration(BaseModel):
    """Runtime configuration settings"""
    data_retention_days: conint(ge=30)
    max_shipment_items: conint(ge=1, le=1000)
    default_currency: CurrencyCode
    temperature_units: str = "CELSIUS"
    distance_units: str = "KILOMETERS"

class FeatureFlags(BaseModel):
    """Feature toggle configuration"""
    enable_customs_check: bool = True
    enable_sensor_alerts: bool = True
    require_two_factor: bool = False
    audit_log_enabled: bool = True

# ====================
# Webhook Models
# ====================

class WebhookSubscription(BaseModel):
    """Event webhook configuration"""
    url: str
    secret: str
    event_types: List[SystemEventType]
    active: bool = True

class WebhookPayload(BaseModel):
    """Webhook event payload"""
    event_id: str
    event_type: SystemEventType
    timestamp: datetime
    resource_type: str
    resource_id: str
    payload: Dict[str, Any]
    signature: str
