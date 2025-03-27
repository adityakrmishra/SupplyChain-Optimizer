"""
Supply Chain Optimizer API Main Entry Point
-------------------------------------------
Production-ready implementation with:
- JWT Authentication
- Role-Based Access Control
- Distributed Tracing
- Rate Limiting
- Async Database Operations
- AI/ML Integration
- Advanced Error Handling
- Comprehensive Documentation
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, List, Dict, Optional

import aioredis
import psutil
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    Request,
    BackgroundTasks,
    Security,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from fastapi.staticfiles import StaticFiles
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse

from .config import settings
from .database import (
    async_session,
    engine,
    Base,
    get_db,
    redis_connection,
)
from .models import schemas
from .models.models import (
    User,
    Shipment,
    InventoryItem,
    Supplier,
    RouteOptimizationResult,
)
from .routers import (
    shipments,
    inventory,
    suppliers,
    analytics,
    ai_ml,
)
from .schemas.common import APIResponse, HealthCheckResponse
from .services import (
    authentication,
    authorization,
    notifications,
    geo_services,
    optimization_engine,
)
from .utils import (
    error_handlers,
    logging_config,
    security_headers,
    request_id,
    cache_control,
)

# Initialize logging
logging_config.setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup events
    logger.info("Starting application initialization")
    
    # Initialize database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize Redis
    app.state.redis = await aioredis.from_url(settings.REDIS_URL)
    await FastAPILimiter.init(app.state.redis)
    
    # Warmup AI models
    await optimization_engine.warmup_models()
    
    logger.info("Application initialization complete")
    yield
    
    # Shutdown events
    logger.info("Starting application shutdown")
    await engine.dispose()
    await app.state.redis.close()
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Supply Chain Optimizer API",
    version="2.3.0",
    description=(
        "Enterprise-grade supply chain management system with AI/ML capabilities\n\n"
        "**Key Features:**\n"
        "- Real-time shipment tracking\n"
        "- Predictive route optimization\n"
        - Inventory demand forecasting\n"
        - Supplier risk analysis\n"
        - Blockchain integration\n"
        - IoT sensor monitoring"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ======================
# Middleware Configuration
# ======================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
app.add_middleware(request_id.RequestIDMiddleware)
app.add_middleware(security_headers.SecurityHeadersMiddleware)
app.add_middleware(cache_control.CacheControlMiddleware)

# ======================
# Database & Service Initialization
# ======================

@app.on_event("startup")
async def startup_db():
    """Initialize database connection pool"""
    try:
        await engine.connect()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

# ======================
# Exception Handlers
# ======================

app.add_exception_handler(HTTPException, error_handlers.http_exception_handler)
app.add_exception_handler(SQLAlchemyError, error_handlers.database_exception_handler)
app.add_exception_handler(ValidationError, error_handlers.validation_exception_handler)
app.add_exception_handler(Exception, error_handlers.generic_exception_handler)

# ======================
# Security Configuration
# ======================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/v1/auth/token",
    scopes={
        "shipments:read": "Read shipment information",
        "shipments:write": "Create/update shipments",
        "admin": "Full administrative access",
    },
)

# ======================
# Router Imports
# ======================

app.include_router(
    shipments.router,
    prefix="/v1/shipments",
    tags=["Shipments"],
    dependencies=[Depends(RateLimiter(times=100, minutes=1))],
)

app.include_router(
    inventory.router,
    prefix="/v1/inventory",
    tags=["Inventory"],
    dependencies=[Depends(RateLimiter(times=200, minutes=1))],
)

app.include_router(
    suppliers.router,
    prefix="/v1/suppliers",
    tags=["Suppliers"],
    dependencies=[Depends(RateLimiter(times=150, minutes=1))],
)

app.include_router(
    analytics.router,
    prefix="/v1/analytics",
    tags=["Analytics"],
    dependencies=[Depends(RateLimiter(times=50, minutes=1))],
)

app.include_router(
    ai_ml.router,
    prefix="/v1/ai",
    tags=["AI/ML"],
    dependencies=[Depends(RateLimiter(times=30, minutes=1))],
)

# ======================
# Core API Endpoints
# ======================

@app.post(
    "/v1/auth/token",
    response_model=schemas.Token,
    summary="Authenticate User",
    description="Obtain JWT tokens for API access",
    tags=["Authentication"],
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db),
) -> schemas.Token:
    """Authenticate user and generate JWT tokens"""
    user = await authentication.authenticate_user(
        db, form_data.username, form_data.password
    )
    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = authentication.create_access_token(
        data={"sub": user.email, "scopes": form_data.scopes}
    )
    refresh_token = authentication.create_refresh_token(data={"sub": user.email})
    
    await redis_connection.set(
        f"refresh_token:{user.email}",
        refresh_token,
        ex=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
    )
    
    logger.info(f"User {user.email} logged in successfully")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
    }

@app.post(
    "/v1/auth/refresh",
    response_model=schemas.Token,
    summary="Refresh Access Token",
    tags=["Authentication"],
)
async def refresh_access_token(
    refresh_token: str, db=Depends(get_db)
) -> schemas.Token:
    """Refresh expired access tokens using refresh token"""
    email = authentication.verify_refresh_token(refresh_token)
    stored_token = await redis_connection.get(f"refresh_token:{email}")
    
    if stored_token != refresh_token:
        logger.warning(f"Invalid refresh token attempt for user: {email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    new_access_token = authentication.create_access_token(
        data={"sub": email, "scopes": []}
    )
    new_refresh_token = authentication.create_refresh_token(data={"sub": email})
    
    await redis_connection.set(
        f"refresh_token:{email}",
        new_refresh_token,
        ex=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
    )
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "refresh_token": new_refresh_token,
    }

@app.get(
    "/v1/system/health",
    response_model=HealthCheckResponse,
    summary="System Health Check",
    tags=["System"],
)
async def health_check() -> HealthCheckResponse:
    """Check system health and component status"""
    return {
        "status": "OK",
        "version": settings.VERSION,
        "database_status": "CONNECTED" if engine.is_connected else "DISCONNECTED",
        "cache_status": "OK" if app.state.redis.ping() else "ERROR",
        "uptime": timedelta(seconds=time.time() - psutil.Process().create_time()),
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "cpu_usage": f"{psutil.cpu_percent()}%",
    }

@app.post(
    "/v1/shipments/{shipment_id}/track",
    response_model=schemas.ShipmentResponse,
    summary="Update Shipment Tracking",
    tags=["Shipments"],
    dependencies=[
        Depends(RateLimiter(times=50, minutes=1)),
        Security(authorization.verify_scopes, scopes=["shipments:write"]),
    ],
)
async def track_shipment(
    shipment_id: str,
    update: schemas.ShipmentTrackingUpdate,
    background_tasks: BackgroundTasks,
    current_user: schemas.UserResponse = Depends(
        authorization.get_current_active_user
    ),
    db=Depends(get_db),
) -> schemas.ShipmentResponse:
    """Update shipment tracking information with IoT data"""
    try:
        shipment = await Shipment.get(db, shipment_id)
        if not shipment:
            logger.error(f"Shipment {shipment_id} not found")
            raise HTTPException(status_code=404, detail="Shipment not found")
        
        # Process geospatial data
        if update.location:
            geo_data = await geo_services.reverse_geocode(
                update.location.lat,
                update.location.lng,
            )
            update.location_data = geo_data
        
        updated_shipment = await Shipment.update(db, shipment_id, update)
        
        # Trigger background tasks
        background_tasks.add_task(
            notifications.send_shipment_update,
            updated_shipment,
            current_user.email,
        )
        background_tasks.add_task(
            optimization_engine.analyze_route_deviation,
            updated_shipment,
        )
        
        logger.info(f"Shipment {shipment_id} updated by {current_user.email}")
        return updated_shipment
    
    except SQLAlchemyError as e:
        logger.error(f"Database error updating shipment: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error updating shipment tracking data"
        )

@app.get(
    "/v1/optimize-routes",
    response_model=schemas.RouteOptimizationResult,
    summary="Optimize Delivery Routes",
    tags=["AI/ML"],
    dependencies=[
        Depends(RateLimiter(times=10, minutes=1)),
        Security(authorization.verify_scopes, scopes=["shipments:read"]),
    ],
)
async def optimize_delivery_routes(
    origin: str,
    destination: str,
    constraints: List[schemas.OptimizationConstraint] = ["cost", "emissions"],
    db=Depends(get_db),
) -> schemas.RouteOptimizationResult:
    """Calculate optimal delivery routes using ML model"""
    try:
        start = time.perf_counter()
        
        # Get real-time constraints
        traffic_data = await geo_services.get_live_traffic(origin, destination)
        weather_data = await geo_services.get_weather_forecast(origin)
        
        # Prepare optimization parameters
        params = {
            "origin": origin,
            "destination": destination,
            "constraints": constraints,
            "traffic_conditions": traffic_data,
            "weather_conditions": weather_data,
            "vehicle_type": "TRUCK",
        }
        
        # Execute optimization
        result = await optimization_engine.optimize_route(params)
        
        # Save result to database
        optimization_record = await RouteOptimizationResult.create(
            db,
            {
                "parameters": params,
                "result": result.dict(),
                "processing_time": time.perf_counter() - start,
            },
        )
        
        logger.info(
            f"Route optimized from {origin} to {destination} in "
            f"{optimization_record.processing_time:.2f}s"
        )
        return optimization_record
    
    except optimization_engine.OptimizationError as e:
        logger.error(f"Route optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Route optimization failed",
            headers={"Retry-After": "30"},
        )

@app.post(
    "/v1/inventory/predict-demand",
    response_model=schemas.DemandForecast,
    summary="Predict Inventory Demand",
    tags=["AI/ML"],
    dependencies=[
        Depends(RateLimiter(times=20, minutes=1)),
        Security(authorization.verify_scopes, scopes=["inventory:read"]),
    ],
)
async def predict_inventory_demand(
    forecast_params: schemas.DemandForecastParams,
    db=Depends(get_db),
) -> schemas.DemandForecast:
    """Predict future inventory requirements using ML"""
    try:
        # Retrieve historical data
        history = await InventoryItem.get_sales_history(
            db,
            forecast_params.product_id,
            forecast_params.history_window_days,
        )
        
        # Generate forecast
        forecast = await optimization_engine.predict_demand(
            product_id=forecast_params.product_id,
            historical_data=history,
            lookahead_days=forecast_params.lookahead_days,
        )
        
        # Save forecast results
        await InventoryItem.save_forecast(db, forecast)
        
        return forecast
    
    except optimization_engine.PredictionError as e:
        logger.error(f"Demand prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Demand prediction error",
            headers={"Retry-After": "60"},
        )

@app.get(
    "/v1/suppliers/risk-assessment/{supplier_id}",
    response_model=schemas.SupplierRiskAssessment,
    summary="Assess Supplier Risk",
    tags=["Suppliers"],
    dependencies=[
        Depends(RateLimiter(times=15, minutes=1)),
        Security(authorization.verify_scopes, scopes=["suppliers:read"]),
    ],
)
async def assess_supplier_risk(
    supplier_id: str,
    db=Depends(get_db),
) -> schemas.SupplierRiskAssessment:
    """Evaluate supplier risk using AI models"""
    try:
        supplier = await Supplier.get(db, supplier_id)
        if not supplier:
            raise HTTPException(status_code=404, detail="Supplier not found")
        
        risk_factors = await optimization_engine.analyze_supplier_risk(
            supplier.dict(),
            await Supplier.get_performance_history(db, supplier_id),
        )
        
        return {
            "supplier_id": supplier_id,
            "risk_score": risk_factors["score"],
            "risk_factors": risk_factors["factors"],
            "recommendation": risk_factors["recommendation"],
        }
    
    except optimization_engine.AnalysisError as e:
        logger.error(f"Risk assessment failed for supplier {supplier_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Risk assessment error")

# ======================
# Advanced Features
# ======================

@app.post(
    "/v1/blockchain/register-shipment",
    response_model=schemas.BlockchainRecord,
    summary="Register Shipment on Blockchain",
    tags=["Blockchain"],
    dependencies=[
        Depends(RateLimiter(times=5, minutes=1)),
        Security(authorization.verify_scopes, scopes=["shipments:write"]),
    ],
)
async def register_shipment_on_blockchain(
    shipment_id: str,
    db=Depends(get_db),
) -> schemas.BlockchainRecord:
    """Create immutable blockchain record for shipment"""
    try:
        shipment = await Shipment.get(db, shipment_id)
        if not shipment:
            raise HTTPException(status_code=404, detail="Shipment not found")
        
        tx_hash = await blockchain_integration.create_shipment_record(
            shipment.dict(),
            settings.BLOCKCHAIN_PRIVATE_KEY,
        )
        
        await Shipment.update(
            db,
            shipment_id,
            {"blockchain_tx_hash": tx_hash},
        )
        
        return {"tx_hash": tx_hash, "shipment_id": shipment_id}
    
    except blockchain_integration.BlockchainError as e:
        logger.error(f"Blockchain registration failed: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail="Blockchain service unavailable",
        )

@app.post(
    "/v1/iot/webhook",
    summary="Receive IoT Device Data",
    tags=["IoT"],
    include_in_schema=False,
)
async def iot_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
):
    """Endpoint for IoT devices to submit sensor data"""
    try:
        payload = await request.json()
        validation_result = await iot_services.validate_webhook_payload(payload)
        
        if not validation_result["valid"]:
            logger.warning(f"Invalid IoT payload: {validation_result['errors']}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid payload format"},
            )
        
        # Add background processing tasks
        background_tasks.add_task(
            iot_services.process_webhook_payload,
            payload,
            db
        )
        
        # Immediate response to acknowledge receipt
        return JSONResponse(
            status_code=202,
            content={
                "detail": "Payload accepted for processing",
                "device_id": payload.get("device_id"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    except json.JSONDecodeError:
        logger.error("Invalid JSON payload received")
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid JSON format"},
        )
        
    except Exception as e:
        logger.error(f"Unexpected error processing IoT webhook: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


# IoT Services Implementation (iot_services.py)
# --------------------------------------------------
async def process_webhook_payload(payload: dict, db: AsyncSession):
    """Process validated IoT payload in background"""
    try:
        logger.info(f"Processing IoT payload from device {payload.get('device_id')}")
        
        # Convert raw payload to standardized format
        sensor_data = await transform_sensor_data(payload)
        
        # Save to time-series database
        await save_sensor_readings(sensor_data)
        
        # Check for anomalies
        anomalies = await detect_data_anomalies(sensor_data)
        
        if anomalies:
            logger.warning(f"Detected anomalies in sensor data: {anomalies}")
            await trigger_alert_notifications(
                sensor_data["shipment_id"],
                anomalies,
                sensor_data["readings"]
            )
            
            # Update shipment status if critical anomaly
            if any(a["severity"] == "CRITICAL" for a in anomalies):
                await update_shipment_status(
                    db,
                    sensor_data["shipment_id"],
                    schemas.ShipmentStatus.ANOMALY_DETECTED,
                    "Critical sensor anomaly detected"
                )

        # Update real-time analytics
        await analytics_engine.update_live_dashboard(sensor_data)

    except Exception as e:
        logger.error(f"Error processing IoT payload: {str(e)}")
        await handle_processing_failure(payload, str(e))


async def transform_sensor_data(raw_data: dict) -> schemas.SensorData:
    """Convert raw IoT payload to standardized schema"""
    return schemas.SensorData(
        device_id=raw_data["device_id"],
        shipment_id=raw_data["metadata"]["shipment_id"],
        timestamp=datetime.fromisoformat(raw_data["timestamp"]),
        readings=[
            schemas.SensorReading(
                sensor_type=reading["type"],
                value=reading["value"],
                unit=reading["unit"],
                coordinates=schemas.GeoLocation(
                    lat=reading["coordinates"]["lat"],
                    lng=reading["coordinates"]["lng"]
                ) if reading.get("coordinates") else None
            ) for reading in raw_data["readings"]
        ],
        battery_level=raw_data.get("battery"),
        signal_strength=raw_data.get("signal_strength")
    )


async def save_sensor_readings(data: schemas.SensorData):
    """Store sensor data in time-series database"""
    try:
        # InfluxDB example
        await influx_client.write(
            bucket=settings.INFLUX_BUCKET,
            record={
                "measurement": "sensor_readings",
                "tags": {
                    "device_id": data.device_id,
                    "shipment_id": data.shipment_id,
                    "sensor_type": reading.sensor_type
                },
                "fields": {
                    "value": reading.value,
                    "battery": data.battery_level,
                    "signal": data.signal_strength
                },
                "time": data.timestamp
            } for reading in data.readings
        )
    except Exception as e:
        logger.error(f"Failed to save sensor data: {str(e)}")
        raise


async def detect_data_anomalies(data: schemas.SensorData) -> List[schemas.Anomaly]:
    """Analyze sensor readings for anomalies using ML models"""
    anomalies = []
    
    for reading in data.readings:
        try:
            model = anomaly_models[reading.sensor_type]
            prediction = await model.predict(reading.value)
            
            if prediction.is_anomaly:
                anomalies.append(schemas.Anomaly(
                    sensor_type=reading.sensor_type,
                    value=reading.value,
                    severity=prediction.severity,
                    description=prediction.description
                ))
                
        except KeyError:
            logger.warning(f"No anomaly model for sensor type {reading.sensor_type}")
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
    
    return anomalies


async def trigger_alert_notifications(shipment_id: str, anomalies: list, readings: list):
    """Send real-time alerts for detected anomalies"""
    try:
        # Send internal system alert
        await notifications.send_system_alert(
            type="SENSOR_ANOMALY",
            message=f"Anomalies detected in shipment {shipment_id}",
            metadata={
                "shipment_id": shipment_id,
                "anomalies": [a.dict() for a in anomalies],
                "readings": [r.dict() for r in readings]
            }
        )

        # Notify responsible parties
        await notifications.notify_shipment_stakeholders(
            shipment_id=shipment_id,
            notification_type=schemas.NotificationType.SENSOR_ANOMALY,
            message="Sensor anomalies detected in shipment",
            urgency="HIGH"
        )

        # Update monitoring dashboard
        await realtime_monitoring.update_alert_feed(
            shipment_id=shipment_id,
            anomalies=anomalies
        )

    except Exception as e:
        logger.error(f"Failed to trigger alerts: {str(e)}")


async def handle_processing_failure(raw_payload: dict, error: str):
    """Handle failed IoT payload processing"""
    try:
        await influx_client.write(
            bucket=settings.INFLUX_BUCKET,
            record={
                "measurement": "processing_errors",
                "tags": {
                    "device_id": raw_payload.get("device_id"),
                    "error_type": "iot_processing"
                },
                "fields": {
                    "error_message": error,
                    "payload": str(raw_payload)
                },
                "time": datetime.utcnow().isoformat()
            }
        )
        
        await notifications.send_system_alert(
            type="IOT_PROCESSING_FAILURE",
            message="Failed to process IoT payload",
            metadata={
                "error": error,
                "device_id": raw_payload.get("device_id"),
                "payload_sample": str(raw_payload)[:500]
            }
        )
    except Exception as e:
        logger.error(f"Failed to log processing error: {str(e)}")
