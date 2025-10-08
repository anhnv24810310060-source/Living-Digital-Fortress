 
# 📡 ShieldX API Reference

## Table of Contents

  - [Overview](https://www.google.com/search?q=%23overview-1)
  - [Authentication](https://www.google.com/search?q=%23authentication)
  - [Rate Limiting](https://www.google.com/search?q=%23rate-limiting)
  - [Error Handling](https://www.google.com/search?q=%23error-handling)
  - [Orchestrator API](https://www.google.com/search?q=%23orchestrator-api)
  - [Ingress API](https://www.google.com/search?q=%23ingress-api)
  - [Credits API](https://www.google.com/search?q=%23credits-api)
  - [Continuous Authentication API](https://www.google.com/search?q=%23continuous-authentication-api)
  - [Guardian API (Sandbox)](https://www.google.com/search?q=%23guardian-api-sandbox)
  - [Policy Rollout API](https://www.google.com/search?q=%23policy-rollout-api)

-----

## Overview

ShieldX provides RESTful APIs for all its services. All APIs use JSON for request and response payloads and follow standard HTTP conventions.

### Base URLs

| Service | Local Base URL | Port |
| :--- | :--- | :--- |
| Orchestrator | `http://localhost:8080` | 8080 |
| Ingress | `http://localhost:8081` | 8081 |
| Guardian | `http://localhost:9090` | 9090 |
| Credits | `http://localhost:5004` | 5004 |
| ContAuth | `http://localhost:5002` | 5002 |

-----

## Authentication

Most endpoints require a JWT Bearer Token for authentication, obtained from the Auth Service.

```bash
curl -H "Authorization: Bearer <your_jwt_token>" \
  http://localhost:8080/api/v1/some-resource
```

-----

## Rate Limiting

API endpoints are rate-limited based on tenant subscription plans. Exceeding the limit will result in a `429 Too Many Requests` error. Check the following headers in the response:

  - `X-RateLimit-Limit`: The maximum number of requests allowed in the window.
  - `X-RateLimit-Remaining`: The number of requests remaining in the current window.
  - `X-RateLimit-Reset`: The Unix timestamp when the limit resets.

-----

## Error Handling

Errors are returned with a standard JSON structure.

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested tenant could not be found.",
    "request_id": "req-1a2b3c4d"
  }
}
```

-----

## Orchestrator API

**Base URL**: `/api/v1`

### Evaluate Request

`POST /evaluate`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "request": {
    "method": "POST",
    "path": "/api/v1/data",
    "ip": "203.0.113.45"
  }
}
```

\</details\>
\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "decision": "allow",
  "risk_score": 0.15,
  "actions": ["log", "monitor"],
  "policy_version": "1.2.3"
}
```

\</details\>

-----

## Ingress API

**Base URL**: `/api/v1`

### Process Ingress Traffic

`POST /ingress/process`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "request_id": "req-789",
  "method": "GET",
  "path": "/api/v1/users",
  "headers": {
    "X-Forwarded-For": "203.0.113.45"
  }
}
```

\</details\>
\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "request_id": "req-789",
  "decision": "allow",
  "risk_score": 0.12,
  "actions": ["log"]
}
```

\</details\>

-----

## Credits API

**Base URL**: `/api/v1`

### Check Balance

`GET /credits/balance/{tenant_id}`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "balance": 50000,
  "currency": "credits",
  "last_updated": "2025-10-08T17:21:00Z"
}
```

\</details\>

### Consume Credits

`POST /credits/consume`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "amount": 100,
  "description": "Sandbox analysis for file: report.exe",
  "metadata": {
    "service": "guardian",
    "analysis_id": "ana-abc-123"
  }
}
```

\</details\>

### Add Credits

`POST /credits/add`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "amount": 10000,
  "description": "Monthly subscription renewal",
  "payment_id": "pay_1a2b3c4d5e6f"
}
```

\</details\>

-----

## Continuous Authentication API

**Base URL**: `/api/v1`

This service tracks user behavior to generate a real-time trust score.

### Get User Trust Score

`GET /contauth/score/{user_id}`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "user_id": "user-456",
  "trust_score": 0.95,
  "last_evaluated": "2025-10-08T17:22:00Z",
  "factors": [
    { "name": "geolocation", "score": 0.98 },
    { "name": "device_fingerprint", "score": 1.0 },
    { "name": "behavioral_biometrics", "score": 0.92 }
  ]
}
```

\</details\>

-----

## Guardian API (Sandbox)

**Base URL**: `/api/v1`

This service provides a secure, isolated environment for analyzing suspicious files and code.

### Analyze File

`POST /guardian/analyze` (Content-Type: `multipart/form-data`)

\<details\>\<summary\>View Response Example (202 Accepted)\</summary\>

```json
{
  "analysis_id": "ana-def-456",
  "file_name": "suspicious.exe",
  "status": "queued"
}
```

\</details\>

### Get Analysis Results

`GET /guardian/analysis/{analysis_id}`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "analysis_id": "ana-def-456",
  "status": "completed",
  "malicious": true,
  "threat_score": 0.88,
  "behaviors": ["Created file in system directory", "Connected to known C2 server"],
  "yara_matches": ["Win32_Trojan_Generic"]
}
```

\</details\>

-----

## Policy Rollout API

**Base URL**: `/api/v1`

This service manages the gradual deployment of new security policies.

### Get Rollout Status

`GET /rollout/status/{policy_id}`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "policy_id": "policy-123",
  "version": "v1.2.4",
  "status": "canary",
  "traffic_percentage": 1,
  "start_time": "2025-10-08T18:00:00Z"
}
```

\</details\>

 