package deception

import (
    "encoding/json"
    "errors"
    "io"
    "net/http"
    "os"
)

type DSL struct {
    HTTP  *HTTPConfig  `json:"http,omitempty"`
    SSH   *SSHConfig   `json:"ssh,omitempty"`
    Redis *RedisConfig `json:"redis,omitempty"`
    ICS   *ICSConfig   `json:"ics,omitempty"`
}

type HTTPConfig struct {
    Banners  []string `json:"banners,omitempty"`
    JitterMs int      `json:"jitter_ms,omitempty"`
}

type SSHConfig struct {
    Banners []string `json:"banners,omitempty"`
}

type RedisConfig struct {
    Banner string `json:"banner,omitempty"`
}

type ICSConfig struct {
    Modbus *ModbusConfig `json:"modbus,omitempty"`
}

type ModbusConfig struct {
    ExceptionCode byte `json:"exception_code,omitempty"`
}

// Counterstrike types
type ReflectionEngine struct {
    mirrors map[string]*AttackerMirror
}

type AttackerMirror struct {
    AttackerIP   string
    Tools        []AttackerTool
    Exploits     []ExploitAttempt
}

type AttackerTool struct {
    Name      string
    Version   string
    Signature []byte
}

type ExploitAttempt struct {
    CVE       string
    Payload   []byte
    Timestamp string
}

type CounterPayload struct {
    Type        string
    OriginalCVE string
    Payload     []byte
}

func NewReflectionEngine() *ReflectionEngine {
    return &ReflectionEngine{
        mirrors: make(map[string]*AttackerMirror),
    }
}

func (re *ReflectionEngine) CreateMirror(attackerIP string) *AttackerMirror {
    mirror := &AttackerMirror{
        AttackerIP: attackerIP,
        Tools:      make([]AttackerTool, 0),
        Exploits:   make([]ExploitAttempt, 0),
    }
    re.mirrors[attackerIP] = mirror
    return mirror
}

func (am *AttackerMirror) ReflectExploit(payload []byte) *CounterPayload {
    exploit := ExploitAttempt{
        CVE:     "CVE-UNKNOWN",
        Payload: payload,
    }
    am.Exploits = append(am.Exploits, exploit)
    
    return &CounterPayload{
        Type:        "reflection",
        OriginalCVE: exploit.CVE,
        Payload:     []byte("REFLECTED_PAYLOAD"),
    }
}

func LoadFromFile(path string) (*DSL, error) {
    if path == "" { return nil, errors.New("empty path") }
    b, err := os.ReadFile(path)
    if err != nil { return nil, err }
    var d DSL
    if err := json.Unmarshal(b, &d); err != nil { return nil, err }
    return &d, nil
}

func LoadFromURL(url string) (*DSL, error) {
    if url == "" { return nil, errors.New("empty url") }
    resp, err := http.Get(url)
    if err != nil { return nil, err }
    defer resp.Body.Close()
    if resp.StatusCode != 200 { return nil, errors.New("bad status") }
    b, err := io.ReadAll(resp.Body)
    if err != nil { return nil, err }
    var d DSL
    if err := json.Unmarshal(b, &d); err != nil { return nil, err }
    return &d, nil
}
