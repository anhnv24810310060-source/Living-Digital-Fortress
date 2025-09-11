package ratls

import (
    "crypto/tls"
    "fmt"
)

// LoadServerTLS loads a TLS config from cert/key PEM paths. If both empty, returns nil.
func LoadServerTLS(certPath, keyPath string) (*tls.Config, error) {
    if certPath == "" || keyPath == "" { return nil, nil }
    crt, err := tls.LoadX509KeyPair(certPath, keyPath)
    if err != nil { return nil, fmt.Errorf("load keypair: %w", err) }
    return &tls.Config{Certificates: []tls.Certificate{crt}}, nil
}



