 
 -----

# 🛡️ ShieldX Sandbox Service (Guardian)

[](https://golang.org)
[](https://firecracker-microvm.github.io/)
[](https://ebpf.io/)
[](https://opensource.org/licenses/Apache-2.0)

**ShieldX Sandbox Service (Guardian)** provides a secure, hardware-level isolated environment for code execution and analysis of suspicious files, allowing for the detection of malware based on their actual behavior.

## 📋 Table of Contents

- [🎯 Overview](https://www.google.com/search?q=%23-t%E1%BB%95ng-quan)
- [🏗️ Architecture](https://www.google.com/search?q=%23%EF%B8%8F-ki%E1%BA%BFn-tr%C3%BAc)
- [🚀 Quick Start](https://www.google.com/search?q=%23-b%E1%BA%AFt-%C4%91%E1%BA%A7u-nhanh)
- [📡 API Documentation](https://www.google.com/search?q=%23-t%C3%A0i-li%E1%BB%87u-api)
- [🔒 Technology Co [Setup](https://www.google.com/search?q=%23-c%C3%B4ng-ngh%E1%BB%87-c%C3%B4-l%E1%BA%ADp)
- [🔍 Malware Analysis](https://www.google.com/search?q=%23-ph%C3%A2n-t%C3%ADch-m%C3%A3-%C4%91%E1%BB%99c)
- [💻 Development Guide](https://www.google.com/search?q=%23-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-ph%C3%A1t-tri%E1%BB%83n)
- [📊 Monitoring Sat](https://www.google.com/search?q=%23-gi%C3%A1m-s%C3%A1t)
- [🔧 Troubleshooting](https://www.google.com/search?q=%23-x%E1%BB%AD-l%C3%BD-s%E1%BB%B1-c%E1%BB%91)
- [🤝 Contribution & License](https://www.google.com/search?q=%23-%C4%91%C3%B3ng-g%C3%B3p--gi%E1%BA%A5y-ph%C3%A9p)

-----

## 🎯 Overview

### Key Features

- **MicroVM Isolation**: Use Firecracker MicroVMs to create a completely isolated environment at the hardware level via KVM.

- **Kernel Monitoring with eBPF**: Tracks system calls (syscalls), file accesses, and network activity at the kernel level in real time.

- **Automatic Malware Analysis**: Securely execute files and source code to analyze behavior and detect malicious activities.

- **Network Isolation**: Each sandbox runs in a separate network namespace with customizable firewall rules.

- **Resource Limits**: Impose strict CPU, memory, and disk limits on each sandbox using cgroups.

- **Multi-Language Support**: Executes source code for Python, JavaScript, Go, Bash, Ruby, and many other languages.

- **Priority Queue Management**: Arranges execution tasks based on priority (low, normal, high).

### Technologies Used

| Components | Technologies | Versions |
| :--- | :--- | :--- |
| Language | Go | 1.25+ |
| Virtualization | Firecracker | 1.0+ |
| Monitoring | eBPF (BCC/libbpf) | Kernel 4.14+ |
| Containers | Docker | 24.0+ |
| Databases | PostgreSQL | 15+ |
| Queues & Cache | Redis | 7+ |
| Storage | MinIO / AWS S3 | - |

-----

## 🏗️ Architecture

### System Architecture

```plaintext
┌─ ... │
├─ ... │
├─ ... Pre-warmed VMs │
├─ ... File│
├────────────────────────────── ───────────────────────────────┤
│ Isolation Layer │
│ - Firecracker MicroVMs, Network Namespaces, cgroups, seccomp│
├─ ... │
├────────────────────────────── ───────────────────────────────┤
│ Data Layer │
│ - PostgreSQL (Logs, Results), Redis (Queue), S3 (Artifacts) │
└─ ... MicroVM available from Pool
↓
Configure VM (Resources, Network) & Load source code/files
↓
Start Monitoring with eBPF
↓
Execute
(Monitor Syscalls, Network, Files, Processes)
↓
Collect Results & Analyze Behavior
↓
Store Logs & Return VM to Pool
↓
Send Feedback to Client
```

-----

## 🚀 Quick Start

### Required

- **System**: Linux Kernel 4.14+, KVM support, 16GB+ RAM, root privileges.
- **Software**: Go 1.25+, Firecracker 1.0+, Docker 24.0+, BCC tools, YARA.

### Installation Guide
 The installation requires root privileges and low system configuration.

```bash
# 1. Check system requirements (KVM and Kernel)
lsmod | grep kvm
uname -r

# 2. Install Firecracker
# (Follow the official Firecracker documentation)
# Example:
FIRECRACKER_VERSION="v1.7.0"
wget "https://github.com/firecracker-microvm/firecracker/releases/download/${FIRECRACKER_VERSION}/firecracker-${FIRECRACKER_VERSION}-x86_64.tgz"
tar -xzf firecracker-*.tgz
sudo mv release-*/firecracker-* /usr/local/bin/firecracker
firecracker --version

# 3. Install BCC Tools (eBPF) for Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y bpfcc-tools linux-headers-$(uname -r)

# 4. Install YARA
sudo apt-get install -y yara

# 5. Clone a repository and install dependencies
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-sandbox
go mod download

# 6. Start background services using Docker
docker run -d --name shieldx-postgres ...
docker run -d --name shieldx-redis ...
docker run -d --name shieldx-minio ...
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/sandbox-artifacts

# 7. Prepare VM Images (Rootfs and Kernel)
# (Download and place in configured directory)
sudo mkdir -p /var/lib/firecracker/images
# ... download ubuntu-20.04.ext4 and vmlinux ...

# 8. Configure environment variables (create .env file)
# (Copy contents (from .env.example file)

# 9. Run database migrations
migrate -path ./migrations -database "..." up

# 10. Compile eBPF programs
cd ebpf/ && make && cd ..

# 11. Download YARA rules
sudo git clone https://github.com/Yara-Rules/rules.git /etc/yara/rules

# 12. Build and run the service (root privileges required)
go build -o bin/shieldx-sandbox cmd/server/main.go
sudo ./bin/shieldx-sandbox

# 13. Test
curl http://localhost:9090/health
```

-----

## 📡 API Documentation

**Base URL**: `http://localhost:9090/api/v1`
**Authentication**: `Authorization: Bearer <token>`

### Execute Source Code

#### 1\. Enforcement Requirements

`POST /api/v1/sandbox/execute`

\<details\>\<summary\>See Sample Request\</summary\>

```json
{ 
"code": "import os\nprint(os.listdir('/'))", 
"language": "python", 
"timeout": 30, 
"network_enabled": false, 
"priority": "normal"
}
```

\</details\>
\<details\>\<summary\>See Sample Response (202 Accepted)\</summary\>

```json
{ 
"execution_id": "exec-550e8400-e29b-41d4-a716-446655440000", 
"status": "queued"
}
```

\</details\>

#### 2\. Get Execution Results

`GET /api/v1/sandbox/executions/{execution_id}`

\<details\>\<summary\>View Sample Response (200 OK)\</summary\>

```json
{ 
"execution_id": "exec-550e8400-e29b-41d4-a716-446655440000", 
"status": "completed", 
"result": { 
"stdout": "['bin', 'boot', 'dev', 'etc', ...]\n", 
"stderr": "", 
"exit_code": 0 
}, 
"monitoring": { 
"syscalls_count": 150, 
"threat_score": 0.05
}
}
```

\</details\>

### Analyze File

#### 1\. Analyze Request

`POST /api/v1/sandbox/analyze` (Content-Type: `multipart/form-data`)

\<details\>\<summary\>View Sample Response (202 Accepted)\</summary\>

```json
{
"analysis_id": "ana-660e8400-e29b-41d4-a716-446655440001",
"file_name": "suspicious.exe",
"status": "queued"
}
```

\</details\>

#### 2\. Get Analysis Results

`GET /api/v1/sandbox/analysis/{analysis_id}`

\<details\>\<summary\>View Sample Response (200 OK)\</summary\>

```json
{ 
"analysis_id": "ana-660e8400-e29b-41d4-a716-446655440001", 
"status": "completed", 
"malicious": true, 
"threat_level": "high", 
"threat_score": 0.85, 
"behaviors": [ 
{ 
"type": "persistence", 
"description": "Created scheduled task for persistence", 
"severity": "high" 
}, 
{ 
"type": "network_connection", 
"description": "Connected to suspicious IP: 203.0.113.45:443 (Known C2 server)", 
"severity": "critical" 
} 
], 
"yara_matches": [ 
{ 
"rule": "Win32_Trojan_Generic", 
"description": "Generic trojan pattern detected" 
} 
], 
"mitre_attack": [ 
{ 
"technique": "T1055", 
"name": "Process Injection", 
"tactic": "Defense Evasion" 
} 
]
}
```

\</details\>

----- ## 🔒 Isolation Technology

#### Firecracker configuration

```json
{ 
"boot-source": { 
"kernel_image_path": "/var/lib/firecracker/images/vmlinux", 
"boot_args": "console=ttyS0 reboot=k panic=1 pci=off" 
}, 
"drives": [ 
{ 
"drive_id": "rootfs", 
"path_on_host": "/var/lib/firecracker/images/ubuntu-20.04.ext4", 
"is_root_device": true, 
"is_read_only": false 
} 
], 
"machine-config": { 
"vcpu_count": 1, 
"mem_size_mib": 512 
}
}
```

#### Monitor Syscall using eBPF

```c
// ebpf/syscall_monitor.c
#include <uapi/linux/ptrace.h>

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
// Logic to collect syscall information
// and send to user-space via perf buffer.
return 0;
}
```

#### Network Isolation

```go
// internal/sandbox/network.go
func CreateNetworkNamespace(vmID string) error { 
ns := fmt.Sprintf("sandbox-%s", vmID) 
// 1. Create network namespace: ip netns add <ns> 
// 2. Create veth pair: ip link add <veth> type veth peer name <vpeer> 
// 3. Move vpeer to namespace: ip link set <vpeer> netns <ns> 
// 4. Configure IP and enable interfaces 
return nil
}

func DeleteNetworkNamespace(vmID string) error { 
ns := fmt.Sprintf("sandbox-%s", vmID) 
cmd := exec.Command("ip", "netns", "del", ns) 
return cmd.Run()
}
```

#### Resource Limits (cgroups)

```go
// internal/sandbox/cgroups.go
func ApplyResourceLimits(vmPID int, limits ResourceLimits) error {
// Logic to create cgroup and write values
// limits to files like cpu.max, memory.max
// and add VM PID to cgroup.procs file
return nil
}
```

-----

## 🔍 Malware Analysis

The malware analysis flow includes many stages from static analysis to dynamic analysis and risk scoring.

```go
// internal/analysis/behavior_detector.go
var DetectionRules = []DetectionRule{ 
{ 
Name: "file_creation_in_system_dir", 
Description: "A file was created in a sensitive system directory.", 
Severity: "high", 
Pattern: func(log *ExecutionLog) bool { 
// Logic to check created files 
return false 
}, 
}, 
{ 
Name: "suspicious_network_connection", 
Description: "Connection established to a known malicious IP.", 
Severity: "critical", 
Pattern: func(log *ExecutionLog) bool { 
// Logic checks the destination IP against the threat intelligence list 
return false 
}, 
},
}
```

-----

## 💻 Development Guide

### Project Structure Án

```
shieldx-sandbox/
├── cmd/server/main.go
├── internal/
│ ├── api/ # Handlers, routes, middleware
│ ├── sandbox/ # Core logic about Firecracker, Orchestrator, Pool
│ ├── monitoring/ # eBPF monitoring components
│ ├── analysis/ # Analysis components (YARA, Behavior)
│ ├── repository/ # Interaction with DB, Redis, S3
│ └── models/ # Data structures
├── ebpf/ # C source code for eBPF programs
└── go.mod
```

### Managing MicroVM Pools

```go
// internal/sandbox/pool.go
type VMPool struct {
available chan *Sandbox
inUse map[string]*Sandbox
}

func (p *VMPool) Acquire() (*Sandbox, error) {
// Get a VM from the available channel
// Move it to the inUse map
return vm, nil
}

func (p *VMPool) Release(vm *Sandbox) error {
// Reset the state of the VM
// Move it from the inUse map back to the available channel
return nil
}
```

-----

## 📊 Monitoring

Service provides metrics according to the Prometheus standard.

```
# Execution
shieldx_sandbox_executions_total{language,status}
shieldx_sandbox_execution_duration_seconds{language}

# VM Pool
shieldx_sandbox_vms_total{status} # available, in_use

# Analysis
shieldx_sandbox_analyses_total{result} # malicious, benign, error
shieldx_sandbox_threat_score

# Queue
shieldx_sandbox_queue_length
shieldx_sandbox_queue_wait_time_seconds
```

-----

## 🔧 Troubleshooting

#### VM fails to start

- **Check**: `lsmod | grep kvm`, `ls -l /dev/kvm`, user access rights.
- **Solution**: `sudo modprobe kvm_intel`, `sudo usermod -aG kvm $USER`.

#### eBPF monitoring not working

- **Test**: `uname -r` (kernel \>= 4.14), `bpftrace --version` (BCC tools installed).

- **Solution**: Install `linux-headers` corresponding to kernel version, recompile eBPF program.

#### Slow execution or long queue

- **Test**: Metric `shieldx_sandbox_queue_length`, system resources (`top`, `vmstat`).

- **Solution**: Increase VM pool size (`VM_POOL_SIZE`), increase number of workers, or upgrade hardware.

-----

## 🤝 Contributions & Licenses

- **Contributions**: Please refer to the file `CONTRIBUTING.md`.

- **License**: This project is licensed under [Apache License 2.0](https://www.google.com/search?q=LICENSE).