"""
Advanced Carbon Footprint Calculator for Logistics
-------------------------------------------------
Features:
- Multi-modal emission calculations
- Real-time emission factor updates
- Historical data tracking
- Scenario comparison
- Regulatory compliance checks
- Supply chain emission analysis
"""

from typing import Literal, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validate_arguments, validator, root_validator
import requests
import numpy as np
from enum import Enum
import json
from pathlib import Path

# Types
VehicleType = Literal[
    'diesel_truck', 'electric_truck', 'cargo_ship', 
    'air_freight', 'rail', 'drone', 'hydrogen_truck'
]

RegionCode = Literal['EU', 'NA', 'AS', 'SA', 'OC', 'AF']

class EmissionFactorSource(Enum):
    DEFAULT = "default"
    ECOINVENT = "ecoinvent"
    EPA = "epa"
    CUSTOM = "custom"

class CarbonRegulation(Enum):
    EU_ETS = "European Union Emissions Trading System"
    CBP = "California's Cap-and-Trade Program"
    UNFCCC = "UN Climate Change Framework"

class VehicleSpecs(BaseModel):
    max_load_kg: float = Field(..., gt=0)
    fuel_efficiency: Optional[float]  # km/L or km/kWh
    energy_density: Optional[float]  # MJ/L for fuels
    empty_weight_kg: float = 0
    load_efficiency: float = Field(1.0, ge=0, le=1)

class EmissionFactors(BaseModel):
    diesel_truck: float = 0.162  # kg CO2 per ton-km
    electric_truck: float = 0.045
    cargo_ship: float = 0.010
    air_freight: float = 0.805
    rail: float = 0.024
    drone: float = 0.150
    hydrogen_truck: float = 0.100
    source: EmissionFactorSource = EmissionFactorSource.DEFAULT
    region: RegionCode = 'EU'
    last_updated: datetime = Field(default_factory=datetime.now)

    @validator('*', pre=True)
    def validate_emissions(cls, v, field):
        if field.name not in ['source', 'region', 'last_updated'] and v < 0:
            raise ValueError(f"Emission factor {field.name} cannot be negative")
        return v

class CarbonConfig(BaseModel):
    include_well_to_tank: bool = True
    electricity_mix: Dict[RegionCode, float] = Field(
        default={'EU': 0.276, 'NA': 0.385},  # kg CO2/kWh
        description="Regional electricity carbon intensity"
    )
    fuel_carbon_content: Dict[str, float] = Field(
        default={
            'diesel': 2.68,  # kg CO2/L
            'hydrogen': 0.0,
            'jet_fuel': 2.53
        }
    )
    regulations: List[CarbonRegulation] = [CarbonRegulation.EU_ETS]

class LogisticsRoute(BaseModel):
    segments: List[Dict[str, float]] = Field(..., min_items=1)
    vehicle_types: List[VehicleType]
    load_kg: List[float]
    empty_return: List[bool]
    region_codes: List[RegionCode]

class CarbonCalculator:
    def __init__(self, config: CarbonConfig = CarbonConfig()):
        self.factors = EmissionFactors()
        self.config = config
        self.historical_data = []
        self._load_default_factors()
        self._api_session = requests.Session()
        self._api_session.headers.update({'Accept': 'application/json'})

    def _load_default_factors(self):
        """Load emission factors from JSON file"""
        try:
            path = Path(__file__).parent / 'data/emission_factors.json'
            with open(path) as f:
                data = json.load(f)
                self.factors = EmissionFactors(**data)
        except FileNotFoundError:
            pass

    def update_factors(self, source: EmissionFactorSource = None):
        """Refresh emission factors from external sources"""
        if source == EmissionFactorSource.ECOINVENT:
            response = self._api_session.get(
                "https://api.ecoinvent.org/v3/factors",
                params={"type": "transport"}
            )
            response.raise_for_status()
            self.factors = EmissionFactors(**response.json())
        elif source == EmissionFactorSource.EPA:
            response = self._api_session.get(
                "https://www.epa.gov/api/emission-factors"
            )
            response.raise_for_status()
            self.factors = EmissionFactors(**response.json())
        else:
            raise ValueError("Unsupported emission factor source")

    @validate_arguments
    def calculate_co2(
        self,
        distance_km: float,
        vehicle_type: VehicleType,
        load_kg: float = 1000,
        empty_return: bool = False,
        fuel_efficiency: float = None,
        vehicle_specs: VehicleSpecs = None,
        region: RegionCode = 'EU'
    ) -> float:
        """
        Calculate CO2 emissions with advanced parameters
        - distance_km: Total route distance (round trip if empty_return)
        - vehicle_type: Transport mode from VehicleType
        - load_kg: Cargo mass in kilograms
        - empty_return: Include return trip without cargo
        - fuel_efficiency: Custom efficiency override
        - vehicle_specs: Detailed vehicle characteristics
        - region: Regional electricity mix
        """
        factor = self._get_emission_factor(vehicle_type, region)
        
        # Calculate effective load
        if vehicle_specs:
            load_kg = min(load_kg, vehicle_specs.max_load_kg)
            effective_load = load_kg * vehicle_specs.load_efficiency
        else:
            effective_load = load_kg
            
        # Convert to metric tons
        load_tons = effective_load / 1000
        
        # Empty return adjustment
        multiplier = 2 if empty_return else 1
        total_distance = distance_km * multiplier
        
        # Custom fuel efficiency calculation
        if fuel_efficiency and vehicle_type in ['diesel_truck', 'hydrogen_truck']:
            return self._fuel_based_calculation(
                total_distance, 
                vehicle_type, 
                fuel_efficiency,
                load_tons
            )
            
        # Electric vehicle calculation
        if vehicle_type == 'electric_truck':
            return self._electric_calculation(
                total_distance, 
                region,
                vehicle_specs,
                load_tons
            )
            
        # Standard calculation
        emissions = factor * total_distance * load_tons
        
        # Add well-to-tank emissions
        if self.config.include_well_to_tank:
            emissions *= 1.15  # Add 15% for upstream emissions
            
        self._store_calculation({
            "distance": total_distance,
            "emissions": emissions,
            "vehicle": vehicle_type,
            "timestamp": datetime.now()
        })
        
        return round(emissions, 2)

    def _fuel_based_calculation(self, distance: float, vehicle_type: str, 
                              efficiency: float, load_tons: float) -> float:
        """Calculate emissions based on fuel consumption"""
        fuel_type = 'diesel' if vehicle_type == 'diesel_truck' else 'hydrogen'
        fuel_consumption = distance / efficiency
        co2_per_liter = self.config.fuel_carbon_content.get(fuel_type, 0)
        
        emissions = fuel_consumption * co2_per_liter * load_tons
        if vehicle_type == 'hydrogen_truck' and self.config.include_well_to_tank:
            emissions += fuel_consumption * 9.3  # Grey hydrogen production emissions
            
        return emissions

    def _electric_calculation(self, distance: float, region: str, 
                            specs: VehicleSpecs, load_tons: float) -> float:
        """Calculate emissions for electric vehicles"""
        if not specs or not specs.fuel_efficiency:
            raise ValueError("Electric vehicles require fuel_efficiency specification")
            
        energy_consumed = (distance / specs.fuel_efficiency) * load_tons
        carbon_intensity = self.config.electricity_mix.get(region, 0.3)
        
        return energy_consumed * carbon_intensity

    def compare_scenarios(self, scenarios: Dict[str, dict]) -> Dict[str, float]:
        """Compare multiple logistics scenarios"""
        results = {}
        for name, params in scenarios.items():
            try:
                results[name] = self.calculate_co2(**params)
            except Exception as e:
                results[name] = str(e)
        return results

    def optimize_supply_chain(self, routes: List[LogisticsRoute]) -> Dict:
        """Optimize multi-segment supply chain emissions"""
        total_emissions = 0
        breakdown = {}
        
        for route in routes:
            segment_emissions = []
            for i, segment in enumerate(route.segments):
                emissions = self.calculate_co2(
                    distance_km=segment['distance'],
                    vehicle_type=route.vehicle_types[i],
                    load_kg=route.load_kg[i],
                    empty_return=route.empty_return[i],
                    region=route.region_codes[i]
                )
                segment_emissions.append(emissions)
                total_emissions += emissions
                
            breakdown[route.description] = segment_emissions
            
        return {
            "total_kg_co2": round(total_emissions, 2),
            "breakdown": breakdown,
            "equivalent_trees": round(total_emissions / 21.77),  # kg CO2 per tree
            "cost_estimate": total_emissions * self._get_carbon_price()
        }

    def _get_carbon_price(self) -> float:
        """Get current carbon price from regulations"""
        prices = {
            CarbonRegulation.EU_ETS: 85.0,  # EUR/ton
            CarbonRegulation.CBP: 28.0       # USD/ton
        }
        return max(prices.get(reg, 0) for reg in self.config.regulations)

    def get_historical_emissions(self, window_days: int = 30) -> List[dict]:
        """Retrieve historical calculation data"""
        cutoff = datetime.now() - timedelta(days=window_days)
        return [d for d in self.historical_data if d['timestamp'] > cutoff]

    def export_results(self, format: str = 'json') -> str:
        """Export emissions data in specified format"""
        if format == 'json':
            return json.dumps(self.historical_data)
        elif format == 'csv':
            csv = "timestamp,distance,emissions,vehicle\n"
            for entry in self.historical_data:
                csv += f"{entry['timestamp']},{entry['distance']},"\
                       f"{entry['emissions']},{entry['vehicle']}\n"
            return csv
        raise ValueError(f"Unsupported format: {format}")

    def check_compliance(self, emissions: float) -> Dict:
        """Check against configured regulations"""
        limits = {
            CarbonRegulation.EU_ETS: 10000,  # kg CO2/year
            CarbonRegulation.CBP: 25000
        }
        results = {}
        for reg in self.config.regulations:
            limit = limits.get(reg, float('inf'))
            results[reg.value] = {
                "compliant": emissions < limit,
                "allowed": limit,
                "excess": max(0, emissions - limit)
            }
        return results

    def _get_emission_factor(self, vehicle_type: str, region: str) -> float:
        """Get region-adjusted emission factor"""
        base_factor = getattr(self.factors, vehicle_type)
        regional_adjustments = {
            'EU': 1.0,
            'NA': 1.1,
            'AS': 1.2,
            'SA': 1.15,
            'AF': 1.3,
            'OC': 1.05
        }
        return base_factor * regional_adjustments.get(region, 1.0)

    def _store_calculation(self, data: dict):
        """Store calculation in historical data"""
        self.historical_data.append(data)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]

    def plot_emissions_trend(self):
        """Generate emissions visualization"""
        import matplotlib.pyplot as plt
        dates = [d['timestamp'] for d in self.historical_data]
        emissions = [d['emissions'] for d in self.historical_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, emissions, marker='o', linestyle='-')
        plt.title('Carbon Emissions Over Time')
        plt.xlabel('Date')
        plt.ylabel('kg CO2')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def optimize_fleet_mix(self, routes: List[dict]) -> Dict:
        """
        Find optimal vehicle mix for emissions reduction
        - routes: List of route dictionaries with vehicle options
        Returns optimal vehicle assignments and savings
        """
        optimized = {}
        total_savings = 0
        
        for route in routes:
            options = {}
            for vehicle in route['allowed_vehicles']:
                emissions = self.calculate_co2(
                    route['distance'],
                    vehicle,
                    route['load'],
                    route['empty_return']
                )
                options[vehicle] = emissions
                
            best_option = min(options, key=options.get)
            savings = options[route['current_vehicle']] - options[best_option]
            
            optimized[route['id']] = {
                'recommended_vehicle': best_option,
                'emissions_savings': savings,
                'cost_impact': self._get_cost_impact(route, best_option)
            }
            total_savings += savings
            
        return {
            "total_savings_kg": round(total_savings, 2),
            "recommendations": optimized
        }

    def _get_cost_impact(self, route: dict, new_vehicle: str) -> float:
        """Estimate cost difference from vehicle change"""
        current_cost = route['cost_per_km'] * route['distance']
        new_cost = {
            'diesel_truck': 0.35,
            'electric_truck': 0.28,
            'rail': 0.15
        }.get(new_vehicle, 0.3) * route['distance']
        
        return new_cost - current_cost

# Example Usage
if __name__ == "__main__":
    calculator = CarbonCalculator(config=CarbonConfig(regulations=[CarbonRegulation.EU_ETS]))
    
    # Complex logistics scenario
    scenario = {
        "distance_km": 1200,
        "vehicle_type": "diesel_truck",
        "load_kg": 15000,
        "empty_return": True,
        "vehicle_specs": VehicleSpecs(
            max_load_kg=18000,
            fuel_efficiency=3.2,
            load_efficiency=0.85
        )
    }
    
    emissions = calculator.calculate_co2(**scenario)
    print(f"Total Emissions: {emissions} kg CO2")
    
    # Fleet optimization example
    routes = [{
        "id": "route-123",
        "distance": 450,
        "load": 8000,
        "empty_return": False,
        "current_vehicle": "diesel_truck",
        "allowed_vehicles": ["diesel_truck", "electric_truck", "rail"],
        "cost_per_km": 0.40
    }]
    
    optimization = calculator.optimize_fleet_mix(routes)
    print(f"Potential Savings: {optimization['total_savings_kg']} kg CO2")
    
    # Generate compliance report
    compliance = calculator.check_compliance(250000)
    print(f"EU ETS Compliance: {compliance['European Union Emissions Trading System']['compliant']}")
    
    # Export historical data
    print(calculator.export_results(format='csv'))
