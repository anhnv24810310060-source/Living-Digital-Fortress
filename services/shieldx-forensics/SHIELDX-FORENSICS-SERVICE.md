 
-----

# 🛡️ ShieldX Forensics Service

[](https://golang.org)
[](https://www.python.org/)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX Forensics Service** is a centralized system for investigating, analyzing, and responding to cybersecurity incidents. This service provides tools to collect evidence, reconstruct attack timelines, and generate detailed reports.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ System Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  - [🚀 Quick Start](https://www.google.com/search?q=%23-quick-start)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation & Startup](https://www.google.com/search?q=%23installation--startup)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Incident Management](https://www.google.com/search?q=%23incident-management)
      - [Evidence Collection](https://www.google.com/search?q=%23evidence-collection)
      - [Forensic Analysis](https://www.google.com/search?q=%23forensic-analysis)
      - [Timeline Reconstruction](https://www.google.com/search?q=%23timeline-reconstruction)
      - [Report Generation](https://www.google.com/search?q=%23report-generation)
  - [🔍 Analysis Process](https://www.google.com/search?q=%23-analysis-process)
      - [Supported Analysis Types](https://www.google.com/search?q=%23supported-analysis-types)
      - [Chain of Custody](https://www.google.com/search?q=%23chain-of-custody)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [Example of an Analyzer](https://www.google.com/search?q=%23example-of-an-analyzer)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [📚 References](https://www.google.com/search?q=%23-references)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Key Features

  - **Incident Recording**: Stores and manages the entire lifecycle of security incidents.
  - **Evidence Collection**: Automatically collects digital evidence from multiple sources (logs, network traffic, memory dumps).
  - **Timeline Reconstruction**: Reconstructs the sequence of events of an attack from collected evidence.
  - **Artifact Analysis**: Analyzes artifacts such as malicious files, system logs, and network packets.
  - **Chain of Custody**: Ensures the integrity and non-repudiation of digital evidence.
  - **Automated Report Generation**: Exports detailed investigation reports in various formats (PDF, JSON).

### Technology Stack

  - **Language**: Go 1.25+, Python 3.11+ (for analysis scripts)
  - **Framework**: Gin Web Framework
  - **Database**: PostgreSQL 15+
  - **Object Storage**: MinIO / AWS S3
  - **Queue**: Redis 7+
  - **Analysis Tools**: YARA, ClamAV, Volatility Framework

-----

## 🏗️ System Architecture

```plaintext
┌─────────────────────────────────────────────────────┐
│ ShieldX Forensics Service (Port 5006)               │
├─────────────────────────────────────────────────────┤
│ HTTP API Layer                                      │
│ - Incident Management                               │
│ - Evidence Collection                               │
│ - Analysis Requests                                 │
├─────────────────────────────────────────────────────┤
│ Forensic Engine                                     │
│ - Evidence Collector                                │
│ - Timeline Builder                                  │
│ - Artifact Analyzer                                 │
│ - Report Generator                                  │
├─────────────────────────────────────────────────────┤
│ Data Layer                                          │
│ - PostgreSQL (Incidents, metadata)                  │
│ - MinIO/S3 (Artifacts, evidence files)              │
│ - Redis (Analysis queue)                            │
└─────────────────────────────────────────────────────┘
       │                      │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
│  PostgreSQL │        │    MinIO    │        │    Redis    │
└─────────────┘        └─────────────┘        └─────────────┘
```

-----

## 🚀 Quick Start

### Prerequisites

  - Go `1.25` or newer
  - Python `3.11` or newer
  - PostgreSQL `15` or newer
  - MinIO or an AWS S3 account
  - Redis `7` or newer
  - Docker & Docker Compose

### Installation & Startup

```bash
# 1. Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-forensics

# 2. Install dependencies for Go and Python
go mod download
pip install -r requirements.txt

# 3. Start background services (PostgreSQL, MinIO, Redis) using Docker
# PostgreSQL
docker run -d \
  --name shieldx-postgres \
  -e POSTGRES_USER=forensics_user \
  -e POSTGRES_PASSWORD=forensics_pass \
  -e POSTGRES_DB=shieldx_forensics \
  -p 5432:5432 \
  postgres:15-alpine

# MinIO
docker run -d \
  --name shieldx-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Redis
docker run -d \
  --name shieldx-redis \
  -p 6379:6379 \
  redis:7-alpine

# 4. Configure environment variables
export FORENSICS_PORT=5006
export FORENSICS_DB_HOST=localhost
export FORENSICS_S3_ENDPOINT=localhost:9000
export FORENSICS_S3_ACCESS_KEY=minioadmin
export FORENSICS_S3_SECRET_KEY=minioadmin
export FORENSICS_REDIS_HOST=localhost

# 5. Run database migrations
migrate -path ./migrations \
  -database "postgresql://forensics_user:forensics_pass@localhost:5432/shieldx_forensics?sslmode=disable" \
  up

# 6. Build and run the application
go build -o bin/shieldx-forensics cmd/server/main.go
./bin/shieldx-forensics

# 7. Check the service status
# You should receive {"status": "ok"} if successful
curl http://localhost:5006/health
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:5006/api/v1`

### Incident Management

#### 1\. Create New Incident

`POST /api/v1/incidents`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "title": "Suspected Data Breach",
  "severity": "critical",
  "description": "Unusual data exfiltration detected from production database.",
  "detected_at": "2025-10-08T10:00:00Z",
  "source": "waf",
  "initial_indicators": {
    "source_ip": "203.0.113.45",
    "affected_endpoints": ["/api/v1/users", "/api/v1/data"]
  }
}
```

\</details\>
\<details\>
\<summary\>View Response Example (201 Created)\</summary\>

```json
{
  "incident_id": "inc-789",
  "case_number": "CASE-2025-1008-001",
  "title": "Suspected Data Breach",
  "severity": "critical",
  "status": "open",
  "created_at": "2025-10-08T10:00:00Z"
}
```

\</details\>

#### 2\. List Incidents

`GET /api/v1/incidents?status=open&severity=critical`

#### 3\. Get Incident Details

`GET /api/v1/incidents/{incident_id}`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "incident_id": "inc-789",
  "case_number": "CASE-2025-1008-001",
  "title": "Suspected Data Breach",
  "severity": "critical",
  "status": "investigating",
  "timeline_summary": [
    {
      "timestamp": "2025-10-08T10:00:00Z",
      "event": "Incident detected",
      "source": "waf"
    },
    {
      "timestamp": "2025-10-08T10:05:00Z",
      "event": "Evidence collection started",
      "user": "analyst-1"
    }
  ],
  "evidence_count": 5,
  "artifacts_count": 3
}
```

\</details\>

### Evidence Collection

#### 1\. Add New Evidence

`POST /api/v1/evidence`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "incident_id": "inc-789",
  "type": "network_traffic",
  "source": "firewall-logs",
  "description": "Captured packets during the suspected attack window.",
  "metadata": {
    "start_time": "2025-10-08T09:55:00Z",
    "end_time": "2025-10-08T10:05:00Z",
    "packet_count": 15000
  }
}
```

\</details\>

#### 2\. Upload Artifact

`POST /api/v1/artifacts/upload` (Content-Type: `multipart/form-data`)

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "artifact_id": "art-456",
  "incident_id": "inc-789",
  "filename": "suspicious_file.exe",
  "size_bytes": 2048576,
  "hash_md5": "5d41402abc4b2a76b9719d911017c592",
  "hash_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "uploaded_at": "2025-10-08T10:10:00Z",
  "storage_path": "s3://forensics/inc-789/art-456"
}
```

\</details\>

### Forensic Analysis

#### 1\. Request Analysis

`POST /api/v1/analysis`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "artifact_id": "art-456",
  "analysis_type": "malware_scan",
  "tools": ["yara", "clamav"],
  "priority": "high"
}
```

\</details\>
\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "analysis_id": "ana-999",
  "artifact_id": "art-456",
  "status": "queued",
  "submitted_at": "2025-10-08T10:11:00Z"
}
```

\</details\>

#### 2\. Get Analysis Results

`GET /api/v1/analysis/{analysis_id}`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "analysis_id": "ana-999",
  "artifact_id": "art-456",
  "status": "completed",
  "completed_at": "2025-10-08T10:15:00Z",
  "results": {
    "malware_detected": true,
    "threat_type": "trojan",
    "confidence": 0.95,
    "yara_matches": [
      {
        "rule": "Win32_Trojan_Generic",
        "description": "Generic trojan pattern detected in file."
      }
    ],
    "clamav_result": "Win.Trojan.Agent-12345"
  }
}
```

\</details\>

### Timeline Reconstruction

#### 1\. Get Detailed Timeline

`GET /api/v1/incidents/{incident_id}/timeline`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "incident_id": "inc-789",
  "timeline": [
    {
      "timestamp": "2025-10-08T09:50:00Z",
      "event_type": "reconnaissance",
      "description": "Port scanning detected from 203.0.113.45.",
      "evidence_ids": ["ev-001"]
    },
    {
      "timestamp": "2025-10-08T09:55:00Z",
      "event_type": "exploitation",
      "description": "SQL injection attempt on /api/v1/login.",
      "evidence_ids": ["ev-002", "ev-003"]
    },
    {
      "timestamp": "2025-10-08T10:00:00Z",
      "event_type": "data_exfiltration",
      "description": "Large data transfer to C2 server detected.",
      "evidence_ids": ["ev-004", "ev-005"]
    }
  ]
}
```

\</details\>

### Report Generation

#### 1\. Request Report Generation

`POST /api/v1/incidents/{incident_id}/report`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "format": "pdf",
  "include_sections": [
    "executive_summary",
    "timeline",
    "evidence_summary",
    "analysis_results",
    "recommendations"
  ]
}
```

\</details\>

#### 2\. Download Report

`GET /api/v1/reports/{report_id}/download`

-----

## 🔍 Analysis Process

### Supported Analysis Types

  - **Malware Analysis**
      - Static analysis (YARA rules, strings, PE headers)
      - Dynamic analysis (sandbox execution)
  - **Network Traffic Analysis**
      - PCAP file analysis
      - Detection of anomalous traffic patterns
  - **Log Analysis**
      - Log correlation from multiple sources
      - Anomaly detection
  - **Memory Forensics**
      - Memory dump analysis using the Volatility Framework
      - Extraction of processes, network connections, command history

### Chain of Custody

The system ensures the integrity of evidence by recording every action.

```go
// Model to ensure evidence integrity
type ChainOfCustody struct {
    EvidenceID   string      `json:"evidence_id"`
    CollectedBy  string      `json:"collected_by"`
    CollectedAt  time.Time   `json:"collected_at"`
    InitialHash  string      `json:"initial_hash"`
    Transfers    []Transfer  `json:"transfers"`
    IsVerified   bool        `json:"is_verified"`
}

type Transfer struct {
    From        string    `json:"from"`
    To          string    `json:"to"`
    Timestamp   time.Time `json:"timestamp"`
    Reason      string    `json:"reason"`
    Signature   string    `json:"signature"`
}
```

-----

## 💻 Development Guide

### Project Structure

```
shieldx-forensics/
├── cmd/server/main.go
├── internal/
│   ├── api/handlers/
│   │   ├── incident.go
│   │   ├── evidence.go
│   │   └── analysis.go
│   ├── engine/
│   │   ├── collector.go
│   │   ├── analyzer.go
│   │   └── timeline_builder.go
│   ├── analyzers/  // Specific logic for each analysis type
│   │   ├── malware_analyzer.go
│   │   ├── network_analyzer.go
│   │   └── log_analyzer.go
│   ├── repository/
│   └── models/
├── scripts/        // Python/bash scripts for analysis
│   ├── yara_scan.py
│   ├── volatility_analysis.py
│   └── pcap_analysis.py
└── tests/
```

### Example of an Analyzer

```go
package analyzers

// MalwareAnalyzer performs malware scanning on an artifact
type MalwareAnalyzer struct {
    YaraRulesPath string
    ClamAVClient  *clamav.Client
}

// Analyze runs analysis tools on an artifact
func (a *MalwareAnalyzer) Analyze(artifact *models.Artifact) (*models.AnalysisResult, error) {
    result := &models.AnalysisResult{
        ArtifactID: artifact.ID,
        StartedAt:  time.Now(),
    }
    
    // Run YARA scan via Python script
    yaraMatches, err := a.scanWithYara(artifact.StoragePath)
    if err != nil {
        return nil, err
    }
    result.YaraMatches = yaraMatches
    
    // Run ClamAV scan
    clamavResult, err := a.ClamAVClient.ScanFile(artifact.StoragePath)
    if err != nil {
        return nil, err
    }
    result.ClamAVResult = clamavResult
    
    result.CompletedAt = time.Now()
    result.IsMalwareDetected = len(yaraMatches) > 0 || clamavResult.Infected
    
    return result, nil
}
```

-----

## 🧪 Testing

```bash
# Run all tests
go test ./... -v

# Run tests for a specific function
go test ./internal/engine -run TestCollector -v

# Run integration tests
go test ./tests/integration -v
```

-----

## 📊 Monitoring

The service exports metrics in the Prometheus standard for monitoring.

```
shieldx_forensics_incidents_total{severity="critical"}  # Total number of incidents by severity
shieldx_forensics_evidence_collected_total              # Total number of evidence pieces collected
shieldx_forensics_analysis_duration_seconds             # Processing time of an analysis
shieldx_forensics_artifacts_stored_bytes                # Storage used by artifacts
```

-----

## 🔧 Troubleshooting

#### Analysis Job Fails or Gets Stuck

  - **Check**: Review the service logs for errors related to the specific analyzer (e.g., YARA, Volatility).
  - **Verify**: Ensure the artifact was uploaded correctly to MinIO/S3 and is not corrupted.
  - **Solution**: Check the Redis queue for stuck jobs (`LLEN analysis_queue`). Manually inspect the problematic artifact.

#### High Memory/CPU Usage

  - **Check**: Monitor the resource usage of the `shieldx-forensics` process and its child processes (Python scripts).
  - **Verify**: Large artifacts (e.g., memory dumps) can be resource-intensive.
  - **Solution**: Implement resource limits for analysis jobs. Increase the resources of the host machine or distribute jobs across multiple workers.

-----

## 📚 References

  - [Digital Forensics and Incident Response Guide](https://www.google.com/search?q=https://www.sans.org/cyber-security-resources/incident-handlers-handbook)
  - [The Volatility Framework Official Documentation](https://www.volatilityfoundation.org/)
  - [YARA Rules Documentation](https://yara.readthedocs.io/)

-----

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/shieldx-bot/shieldx/blob/main/LICENSE).