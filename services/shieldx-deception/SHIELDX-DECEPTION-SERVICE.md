 
-----

# 🛡️ ShieldX Deception Service

[](https://golang.org)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX Deception Service** is a system that proactively deploys deception technology to detect, analyze, and misdirect cyber attacks in real-time.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ System Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  - [🚀 Quick Start](https://www.google.com/search?q=%23-quick-start)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation & Startup](https://www.google.com/search?q=%23installation--startup)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Camouflage Management](https://www.google.com/search?q=%23camouflage-management)
      - [Honeypot Management](https://www.google.com/search?q=%23honeypot-management)
      - [Decoy Management](https://www.google.com/search?q=%23decoy-management)
      - [Attacker Profiling](https://www.google.com/search?q=%23attacker-profiling)
  - [🎭 Deception Techniques](https://www.google.com/search?q=%23-deception-techniques)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [Creating a Custom Honeypot](https://www.google.com/search?q=%23creating-a-custom-honeypot)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [📚 References](https://www.google.com/search?q=%23-references)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Key Features

  - **Dynamic Honeypots**: Automatically creates and deploys diverse honeypots (SSH, HTTP, FTP, etc.) to attract and trap attackers.
  - **Server Fingerprint Camouflage**: Alters server headers and banners to mimic different technologies (e.g., Nginx, Apache, IIS).
  - **Fake Endpoints**: Creates fake API endpoints (`/.git/config`, `/admin/backup.sql`) to detect reconnaissance activities.
  - **Decoy Credentials**: Scatters fake credentials (API keys, passwords) throughout the system to detect unauthorized access.
  - **Breadcrumb Trails**: Creates false trails in logs and configuration files to lure attackers into traps.
  - **Attacker Profiling**: Automatically collects information about an attacker's IP, techniques, and behavior.

### Technology Stack

  - **Language**: Go 1.25+
  - **Framework**: Gin Web Framework
  - **Database**: PostgreSQL 15+
  - **Cache**: Redis 7+
  - **Containerization**: Docker & Docker Compose

-----

## 🏗️ System Architecture

The diagram below describes the overall architecture of the service:

```plaintext
┌─────────────────────────────────────────────────────┐
│ ShieldX Deception Service (Port 5005)               │
├─────────────────────────────────────────────────────┤
│ HTTP API Layer                                      │
│ - Camouflage API                                    │
│ - Honeypot Management                               │
│ - Decoy Management                                  │
├─────────────────────────────────────────────────────┤
│ Deception Engine                                    │
│ - Fingerprint Camouflage Engine                     │
│ - Honeypot Orchestrator                             │
│ - Decoy Generator                                   │
│ - Attacker Tracker                                  │
├─────────────────────────────────────────────────────┤
│ Data Layer                                          │
│ - PostgreSQL (Configurations, attack logs)          │
│ - Redis (Template cache, session tracking)          │
└─────────────────────────────────────────────────────┘
       │                      │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
│  PostgreSQL │        │    Redis    │        │    Docker   │
│             │        │             │        │ (Honeypots) │
└─────────────┘        └─────────────┘        └─────────────┘
```

-----

## 🚀 Quick Start

### Prerequisites

  - Go `1.25` or newer
  - PostgreSQL `15` or newer
  - Redis `7` or newer
  - Docker & Docker Compose

### Installation & Startup

Follow these steps to install and run the service on a local machine.

```bash
# 1. Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-deception

# 2. Install dependencies
go mod download

# 3. Start PostgreSQL and Redis using Docker
docker run -d \
  --name shieldx-postgres \
  -e POSTGRES_USER=deception_user \
  -e POSTGRES_PASSWORD=deception_pass \
  -e POSTGRES_DB=shieldx_deception \
  -p 5432:5432 \
  postgres:15-alpine

docker run -d \
  --name shieldx-redis \
  -p 6379:6379 \
  redis:7-alpine

# 4. Configure environment variables
export DECEPTION_PORT=5005
export DECEPTION_DB_HOST=localhost
export DECEPTION_REDIS_HOST=localhost
export DOCKER_HOST=unix:///var/run/docker.sock

# 5. Run database migrations
migrate -path ./migrations \
  -database "postgresql://deception_user:deception_pass@localhost:5432/shieldx_deception?sslmode=disable" \
  up

# 6. Build and run the application
go build -o bin/shieldx-deception cmd/server/main.go
./bin/shieldx-deception

# 7. Check the service status
# You should receive {"status": "ok"} if successful
curl http://localhost:5005/health
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:5005/api/v1`

### Camouflage Management

#### 1\. Get Camouflage Template

`GET /api/v1/camouflage/template?server_type=nginx`

\<details\>
\<summary\>View Response Example (200 OK)\</summary\>

```json
{
  "template_id": "tpl-123",
  "server_type": "nginx",
  "headers": {
    "Server": "nginx/1.18.0",
    "X-Powered-By": "PHP/7.4.3"
  },
  "error_pages": {
    "404": "<html>404 Not Found - nginx</html>"
  }
}
```

\</details\>

#### 2\. Apply Camouflage

`POST /api/v1/camouflage/apply`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "target": "api.example.com",
  "camouflage_type": "apache",
  "custom_headers": {
    "X-Custom": "value"
  }
}
```

\</details\>

### Honeypot Management

#### 1\. Create New Honeypot

`POST /api/v1/honeypots`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "name": "SSH Honeypot",
  "type": "ssh",
  "port": 2222,
  "config": {
    "banner": "OpenSSH_7.4",
    "allowed_users": ["admin", "root"],
    "log_level": "verbose"
  }
}
```

\</details\>
\<details\>
\<summary\>View Response Example (201 Created)\</summary\>

```json
{
  "honeypot_id": "hp-456",
  "name": "SSH Honeypot",
  "type": "ssh",
  "status": "running",
  "endpoint": "honeypot-456.shieldx.internal:2222",
  "created_at": "2025-01-20T10:00:00Z"
}
```

\</details\>

#### 2\. List Honeypots

`GET /api/v1/honeypots?tenant_id=tenant-123&status=running`

#### 3\. Get Honeypot Interaction History

`GET /api/v1/honeypots/{honeypot_id}/interactions?limit=100`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "data": [
    {
      "interaction_id": "int-789",
      "honeypot_id": "hp-456",
      "source_ip": "203.0.113.45",
      "timestamp": "2025-01-20T10:05:00Z",
      "action": "login_attempt",
      "details": {
        "username": "admin",
        "password": "password123",
        "success": false
      }
    }
  ]
}
```

\</details\>

#### 4\. Stop a Honeypot

`POST /api/v1/honeypots/{honeypot_id}/stop`

### Decoy Management

#### 1\. Create Fake Endpoint

`POST /api/v1/decoys/endpoints`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "path": "/admin/backup.sql",
  "response_type": "file",
  "response_data": {
    "filename": "backup.sql",
    "size": "15MB",
    "content_type": "application/sql"
  },
  "alert_on_access": true
}
```

\</details\>

#### 2\. Create Decoy Credentials

`POST /api/v1/decoys/credentials`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "type": "api_key",
  "value": "sk_test_fake_key_12345",
  "location": "config.yaml",
  "alert_on_use": true
}
```

\</details\>

#### 3\. List Alerts from Decoys

`GET /api/v1/decoys/alerts?tenant_id=tenant-123&severity=high`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "data": [
    {
      "alert_id": "alert-999",
      "decoy_type": "credential",
      "decoy_id": "decoy-888",
      "triggered_at": "2025-01-20T10:10:00Z",
      "source_ip": "198.51.100.23",
      "severity": "high",
      "details": {
        "credential_used": "sk_test_fake_key_12345",
        "endpoint": "/api/v1/users"
      }
    }
  ]
}
```

\</details\>

### Attacker Profiling

#### 1\. Get Detailed Attacker Profile

`GET /api/v1/attackers/{source_ip}`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "source_ip": "203.0.113.45",
  "first_seen": "2025-01-20T09:00:00Z",
  "last_seen": "2025-01-20T10:10:00Z",
  "interaction_count": 15,
  "honeypots_accessed": ["hp-456", "hp-789"],
  "decoys_triggered": ["decoy-888"],
  "techniques": ["brute_force", "sql_injection"],
  "threat_score": 85,
  "geolocation": {
    "country": "Unknown",
    "city": "Unknown"
  }
}
```

\</details\>

-----

## 🎭 Deception Techniques

#### 1\. Server Fingerprint Camouflage

Change HTTP headers to fake the underlying technology.

```go
// Example: Camouflage as an Apache server
headers := map[string]string{
    "Server": "Apache/2.4.41 (Ubuntu)",
    "X-Powered-By": "PHP/7.4.3",
}
```

#### 2\. Dynamic Honeypots

Instantiate honeypots on demand to simulate real services.

```go
// SSH Honeypot
type SSHHoneypot struct {
    Port   int
    Banner string
    Logger *log.Logger
}

// HTTP Honeypot
type HTTPHoneypot struct {
    Port      int
    FakePaths []string
    Logger    *log.Logger
}
```

#### 3\. Breadcrumbs

Scatter fake information in configuration files or code comments.

```yaml
# config.yaml (fake)
database:
  host: db.internal.example.com
  username: admin
  password: SuperSecret123!  # This is a decoy!
```

#### 4\. Fake API Endpoints

Create fake sensitive endpoints to detect scanning behavior.

  - `/admin/backup.sql`
  - `/api/internal/users`
  - `/.env`
  - `/.git/config`
  - `/phpMyAdmin/`

-----

## 💻 Development Guide

### Project Structure

```
shieldx-deception/
├── cmd/server/main.go
├── internal/
│   ├── api/handlers/
│   │   ├── camouflage.go
│   │   ├── honeypot.go
│   │   └── decoy.go
│   ├── engine/
│   │   ├── camouflage_engine.go
│   │   ├── honeypot_orchestrator.go
│   │   └── decoy_generator.go
│   ├── honeypots/
│   │   ├── ssh_honeypot.go
│   │   ├── http_honeypot.go
│   │   └── ftp_honeypot.go
│   ├── repository/
│   └── models/
├── templates/
│   ├── nginx.json
│   ├── apache.json
│   └── iis.json
└── tests/
```

### Creating a Custom Honeypot

To extend the system, you can create a new honeypot by implementing the basic interface.

```go
package honeypots

import (
    "fmt"
    "net"
)

type CustomHoneypot struct {
    ID     string
    Type   string
    Port   int
    Config map[string]interface{}
}

func (h *CustomHoneypot) Start() error {
    listener, err := net.Listen("tcp", fmt.Sprintf(":%d", h.Port))
    if err != nil {
        return err
    }
    
    go h.acceptConnections(listener)
    return nil
}

func (h *CustomHoneypot) acceptConnections(listener net.Listener) {
    defer listener.Close()
    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }
        go h.handleConnection(conn)
    }
}

func (h *CustomHoneypot) handleConnection(conn net.Conn) {
    // Logic to handle the connection and log the interaction
    conn.Close()
}
```

-----

## 🧪 Testing

The system has a comprehensive suite of unit and integration tests.

```bash
# Run all tests
go test ./... -v

# Run tests for a specific package (e.g., engine)
go test ./internal/engine -v

# Run integration tests
go test ./tests/integration -v
```

**Example of a test case for the SSH Honeypot:**

```go
func TestSSHHoneypot_CaptureLogin(t *testing.T) {
    honeypot := NewSSHHoneypot(2222)
    err := honeypot.Start()
    assert.NoError(t, err)
    
    // Simulate an attack connection
    conn, _ := net.Dial("tcp", "localhost:2222")
    conn.Write([]byte("admin\npassword123\n"))
    
    // Verify that the interaction was logged
    interactions := honeypot.GetInteractions()
    assert.Equal(t, 1, len(interactions))
}
```

-----

## 📊 Monitoring

The service exports metrics in the Prometheus standard for monitoring.

```
shieldx_deception_honeypots_total         # Total number of active honeypots
shieldx_deception_interactions_total      # Total number of logged interactions
shieldx_deception_decoys_triggered_total  # Total number of triggered decoys
shieldx_deception_attackers_unique        # Number of unique attackers
```

-----

## 🔧 Troubleshooting

#### Honeypot Fails to Start

```bash
# 1. Check if the Docker container is running
docker ps | grep honeypot

# 2. View the container logs
docker logs <container_id_or_name>

# 3. Check if the port is already in use
netstat -tuln | grep <honeypot_port>
```

#### Decoy Does Not Trigger an Alert

```bash
# 1. Verify the decoy's configuration
curl http://localhost:5005/api/v1/decoys/{decoy_id}

# 2. Check the alert settings
curl http://localhost:5005/api/v1/decoys/{decoy_id}/config
```

-----

## 📚 References

  - [Deception Technology Guide](https://shieldx.dev/DeceptionGuide)
  - [Honeypot Best Practices](https://shieldx.dev/HoneypotPractices)
  - [ShieldX System Architecture](https://shieldx.dev/SystemArchitecture)

-----

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/shieldx-bot/shieldx/blob/main/LICENSE).