# FastAPI Endpoints

The `SupplyChain-Optimizer` exposes several RESTful API endpoints using the FastAPI framework. Below is a reference for each available endpoint:

## Base URL

http://{host}:{port}/api/v1

## Endpoints

### 1. Health Check

- **Endpoint**: `/health`
- **Method**: GET
- **Description**: Checks the health status of the API.
- **Responses**:
  - `200 OK`: API is healthy.

### 2. Predict Disruptions

- **Endpoint**: `/predict/disruptions`
- **Method**: POST
- **Description**: Predicts potential supply chain disruptions based on input data.
- **Request Body**:
  ```json
  {
    "weather_data": {...},
    "news_data": {...},
    "historical_data": {...}
  }
- **Responses**:
   - 200 OK: Returns prediction results.
     ```bash
     {
     "disruption_risk": "high",
     "confidence": 0.95
     }
  - 400 Bad Request: Invalid input data.

3. **Optimize Route**
- Endpoint: /optimize/route
- Method: POST
- Description: Provides optimized logistics routes considering various factors.
- Request Body:
                ```bash
                   {
                   "origin": "Location A",
                   "destination": "Location B",
                   "constraints": {
                   "time": "shortest",
                   "cost": "lowest",
                   "carbon_footprint": "minimum"
                  }
                }
         ```
  - Responses:
     - 200 OK: Returns optimized route details.
```json
{
  "route": ["Point A", "Point B", "Point C"],
  "total_distance": "150 km",
  "estimated_time": "2 hours",
  "estimated_cost": "$200"
}
```
   - 400 Bad Request: Invalid input data.

4. **Get Historical Data**
- Endpoint: /data/historical
- Method: GET
- Description: Retrieves historical supply chain data.
- Query Parameters:
  - start_date: Start date for data retrieval (format: YYYY-MM-DD).
  - end_date: End date for data retrieval (format: YYYY-MM-DD).
Responses:
  - 200 OK: Returns historical data within the specified date range.

```json
[
  {
    "date": "2025-03-01",
    "event": "Disruption Event 1",
    "impact": "Moderate"
  },
  {
    "date": "2025-03-15",
    "event": "Disruption Event 2",
    "impact": "Severe"
  }
]
```
   - 400 Bad Request: Invalid date format or range.

### Authentication
Some endpoints may require authentication. Please refer to the Authentication Guide for details on how to authenticate API requests.

### Error Handling
The API returns standard HTTP status codes to indicate the success or failure of a request. In case of an error, the response body will contain a JSON object with details about the error.

For more detailed information on each endpoint, including example requests and responses, please refer to the API Usage Guide.

```arduino
 

These documents provide a comprehensive overview of the system's architecture and the available API endpoints, facilitating better understanding and utilization of the `Supply
::contentReference[oaicite:0]{index=0}
 
```
