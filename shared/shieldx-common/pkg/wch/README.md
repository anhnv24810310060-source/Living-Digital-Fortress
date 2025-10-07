# Whisper Channel Protocol (WCH)

Production-grade secure communication channel with QUIC, camouflage, and distributed rate limiting.

## 🎯 Features

### ✅ QUIC/HTTP3 Server
- Production-grade QUIC implementation
- HTTP/3 support
- Configurable timeouts & stream limits
- Connection metrics & monitoring
- Graceful shutdown

### ✅ TLS Fingerprint Camouflage
- Browser fingerprint rotation (Chrome, Firefox, Safari, Edge)
- JA3 signature rotation
- Traffic obfuscation
- Timing jitter
- HTTP traffic mimicry

### ✅ Distributed Rate Limiting
- Redis-backed distributed rate limiter
- 3 algorithms: Fixed Window, Sliding Window, Token Bucket
- Per-client rate limiting
- Burst support
- Rate limit headers (X-RateLimit-*)

### ✅ Session Management
- Ephemeral key exchange (X25519)
- ECDH shared secret derivation
- HKDF key derivation (SHA-256)
- AES-256-GCM encryption
- Session expiration & cleanup
- Rekey support

## 🚀 Quick Start

### Server Setup

```go
package main

import (
    "context"
    "log"
    "shieldx/pkg/wch"
)

func main() {
    ctx := context.Background()
    
    err := wch.SetupWCHServer(ctx, wch.WCHServerConfig{
        Addr:      ":443",
        CertFile:  "server.crt",
        KeyFile:   "server.key",
        RedisAddr: "localhost:6379",
        RateLimit: 100, // requests per minute
    })
    
    if err != nil {
        log.Fatal(err)
    }
}
```

### Client Usage

```go
package main

import (
    "context"
    "crypto/tls"
    "log"
    "shieldx/pkg/wch"
)

func main() {
    // Create QUIC client
    tlsConfig := &tls.Config{
        InsecureSkipVerify: true, // Only for testing!
        NextProtos:         []string{"h3", "shieldx-wch"},
    }
    
    client := wch.NewQUICClient(tlsConfig)
    defer client.Close()
    
    // Generate ephemeral keypair
    clientPriv, clientPub, _ := wch.GenerateClientEphemeral()
    
    // Connect to server
    // ... establish connection ...
    
    // Prepare inner request
    innerReq := wch.InnerRequest{
        Method: "GET",
        Path:   "/api/data",
        Headers: map[string]string{
            "Authorization": "Bearer token",
        },
    }
    
    innerJSON, _ := json.Marshal(innerReq)
    
    // Encrypt with session key
    nonce, ciphertext, _ := wch.Seal(sessionKey, innerJSON)
    
    // Send envelope
    envelope := &wch.Envelope{
        ChannelID:      channelID,
        NonceB64:       wch.MarshalB64(nonce),
        CiphertextB64:  wch.MarshalB64(ciphertext),
    }
    
    response, _ := client.SendEnvelope(ctx, 
        "https://server.example.com/wch/send", envelope)
}
```

## 📊 Components

### 1. QUIC Server (`quic_server.go`)

Production-grade QUIC/HTTP3 server with:
- Configurable connection limits
- Stream management
- Metrics collection
- Middleware support
- Graceful shutdown

**Configuration**:
```go
config := wch.QUICConfig{
    Addr:                ":443",
    TLSConfig:           tlsConfig,
    MaxIdleTimeout:      30 * time.Second,
    MaxIncomingStreams:  100,
    EnableDatagrams:     true,
    KeepAlivePeriod:     10 * time.Second,
    HandshakeIdleTimeout: 10 * time.Second,
}
```

### 2. Camouflage Engine (`camouflage.go`)

TLS fingerprint rotation and traffic obfuscation:

**Profiles**: Chrome, Firefox, Safari, Edge
**Rotation**: Every 5 minutes (configurable)
**Features**:
- Cipher suite rotation
- Curve preference rotation
- User-Agent rotation
- Custom headers
- Timing jitter
- Traffic padding

**Usage**:
```go
camouflage := wch.NewCamouflageEngine(wch.CamouflageConfig{
    RotationPeriod: 5 * time.Minute,
    EnableJA3:      true,
})

// Apply to HTTP response
camouflage.ApplyFingerprint(w, r)

// Get current TLS config
tlsConfig := camouflage.GetTLSConfig()
```

### 3. Rate Limiter (`rate_limiter.go`)

Distributed rate limiting with Redis:

**Algorithms**:
- **Fixed Window**: Simple counter per window
- **Sliding Window**: Precise request tracking with sorted sets
- **Token Bucket**: Smooth rate limiting with burst support

**Usage**:
```go
// Redis-backed (production)
limiter, _ := wch.NewRedisRateLimiter(wch.RateLimiterConfig{
    RedisAddr:  "localhost:6379",
    Limit:      100,
    Window:     1 * time.Minute,
    Algorithm:  "sliding_window",
})

// Check rate limit
allowed, _ := limiter.Allow(ctx, clientIP)

// Get rate limit info
info, _ := limiter.GetInfo(ctx, clientIP)
fmt.Printf("Remaining: %d/%d\n", info.Remaining, info.Limit)

// As middleware
handler := wch.RateLimitMiddleware(limiter)(yourHandler)
```

### 4. Session Manager (`server.go`)

Secure session management with ephemeral keys:

**Features**:
- X25519 key exchange
- ECDH shared secret
- HKDF key derivation
- Session expiration
- Rekey support

**Usage**:
```go
sessionMgr := wch.NewSessionManager(30 * time.Minute)

// Create session
session, _ := sessionMgr.CreateSession(clientPubKey)

// Get session
session, _ := sessionMgr.GetSession(channelID)

// Update activity (extends expiration)
sessionMgr.UpdateSessionActivity(channelID)
```

## 🔒 Security Features

### Cryptography
- **Key Exchange**: X25519 (ECDH)
- **Encryption**: AES-256-GCM
- **Key Derivation**: HKDF-SHA256
- **TLS**: TLS 1.3 only
- **Ciphers**: Modern cipher suites only

### Privacy
- **Fingerprint Rotation**: Mimics different browsers
- **JA3 Rotation**: Changes TLS fingerprint periodically
- **Traffic Obfuscation**: Random padding & timing jitter
- **HTTP Mimicry**: Disguises data as HTTP traffic

### Anti-Abuse
- **Rate Limiting**: Per-client request limits
- **Session Expiration**: Automatic cleanup
- **Connection Limits**: Prevents resource exhaustion

## 📈 Metrics

```bash
curl http://localhost:443/wch/metrics
```

Response:
```json
{
  "connections_total": 1234,
  "envelopes_sent": 5678,
  "envelopes_received": 5670,
  "encryption_errors": 2,
  "decryption_errors": 1,
  "sessions_created": 456,
  "sessions_expired": 123,
  "sessions_active": 333
}
```

## 🧪 Testing

### Generate Test Certificates

```bash
# Generate self-signed cert for testing
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout server.key -out server.crt \
  -days 365 -subj "/CN=localhost"
```

### Test Rate Limiter

```bash
# Should succeed
for i in {1..100}; do
  curl http://localhost:443/wch/send
done

# Should fail with 429
for i in {101..110}; do
  curl http://localhost:443/wch/send
done
```

### Test Camouflage

```bash
# Check fingerprint rotation
for i in {1..10}; do
  curl -v https://localhost:443/wch/connect 2>&1 | grep "User-Agent"
  sleep 30
done
```

## 🎛️ Configuration

### Environment Variables

```bash
# Server
WCH_ADDR=:443
WCH_CERT_FILE=server.crt
WCH_KEY_FILE=server.key

# Redis
REDIS_ADDR=localhost:6379
REDIS_PASSWORD=
REDIS_DB=0

# Rate Limiting
RATE_LIMIT=100
RATE_LIMIT_WINDOW=1m
RATE_LIMIT_ALGORITHM=sliding_window

# Sessions
SESSION_TTL=30m

# Camouflage
CAMOUFLAGE_ROTATION=5m
ENABLE_JA3_ROTATION=true
```

## 🔧 Advanced Usage

### Custom TLS Profile

```go
customProfile := wch.TLSProfile{
    Name: "CustomBrowser",
    CipherSuites: []uint16{
        tls.TLS_AES_128_GCM_SHA256,
        tls.TLS_CHACHA20_POLY1305_SHA256,
    },
    Curves: []tls.CurveID{
        tls.X25519,
        tls.CurveP256,
    },
    UserAgent: "CustomBrowser/1.0",
    Headers: map[string]string{
        "X-Custom-Header": "value",
    },
}

camouflage := wch.NewCamouflageEngine(wch.CamouflageConfig{
    CustomProfiles: []wch.TLSProfile{customProfile},
})
```

### Custom Rate Limiting Algorithm

```go
// Token bucket with burst
limiter, _ := wch.NewRedisRateLimiter(wch.RateLimiterConfig{
    Algorithm:  "token_bucket",
    Limit:      100,  // 100 req/min sustained
    BurstLimit: 200,  // 200 req burst
    Window:     1 * time.Minute,
})
```

### Traffic Obfuscation

```go
// Add random padding
obfuscated := wch.ObfuscateTraffic(data, 100, 1000)

// Mimic HTTP traffic
httpLike := wch.MimicHTTPTraffic(data)

// Add timing jitter
wch.TimingObfuscation(ctx, 10*time.Millisecond, 100*time.Millisecond)
```

## 📚 Protocol Flow

```
1. Client → Server: Connect Request (ephemeral public key)
2. Server → Client: Connect Response (channel ID, server public key)
3. Both: Derive shared secret via ECDH
4. Both: Derive AES-256 key via HKDF
5. Client → Server: Encrypted Envelope (sealed inner request)
6. Server: Decrypt, process, encrypt response
7. Server → Client: Encrypted Envelope (sealed inner response)
8. Client: Decrypt response
```

## 🔐 Best Practices

1. **Use TLS 1.3** - Never use TLS 1.2 or lower
2. **Rotate Keys** - Enable automatic key rotation
3. **Rate Limit** - Always use distributed rate limiting in production
4. **Monitor Metrics** - Track encryption errors and anomalies
5. **Clean Sessions** - Set appropriate session TTL
6. **Use Redis** - For distributed systems, use Redis rate limiter
7. **Enable Camouflage** - Rotate fingerprints to avoid detection
8. **Validate Input** - Always validate envelope structure
9. **Log Carefully** - Never log plaintext or keys
10. **Test Thoroughly** - Test all failure scenarios

## 🐛 Troubleshooting

### Rate Limiting Issues
```go
// Check rate limit info
info, _ := limiter.GetInfo(ctx, clientIP)
log.Printf("Remaining: %d, Resets at: %v", info.Remaining, info.ResetAt)

// Reset if needed
limiter.Reset(ctx, clientIP)
```

### Session Expiration
```go
// Check session status
session, err := sessionMgr.GetSession(channelID)
if err != nil {
    log.Printf("Session error: %v", err)
}

// Extend session
sessionMgr.UpdateSessionActivity(channelID)
```

### QUIC Connection Issues
```go
// Check QUIC metrics
metrics := quicServer.GetMetrics()
log.Printf("Active connections: %d", metrics.ConnectionsActive)
log.Printf("Handshake failures: %d", metrics.HandshakeFailures)
```

## 📄 License

MIT
