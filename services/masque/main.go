package main

import (
    "encoding/binary"
    "encoding/json"
    "io"
    "log"
    "net"
    "net/http"
    "os"
    "strconv"
    "time"

    quic "github.com/quic-go/quic-go"
)

// This is a simplified MASQUE-like UDP relay over QUIC. It exposes a QUIC listener and relays
// UDP datagrams between clients and a configured upstream (sinkhole/internal).

type udpTarget struct {
    Addr string `json:"addr"`
}

func getenvInt(key string, def int) int { v := os.Getenv(key); if v == "" { return def }; n, err := strconv.Atoi(v); if err != nil { return def }; return n }

func main() {
    addr := os.Getenv("MASQUE_QUIC_ADDR")
    if addr == "" { addr = ":9444" }
    upstream := os.Getenv("MASQUE_UDP_UPSTREAM")
    if upstream == "" { upstream = "127.0.0.1:9" } // discard

    tlsConf := generateInsecureTLSConfig()
    ln, err := quic.ListenAddr(addr, tlsConf, &quic.Config{MaxIncomingStreams: 1024})
    if err != nil { log.Fatalf("listen quic: %v", err) }
    log.Printf("[masque] QUIC listening on %s -> UDP %s", addr, upstream)

    // Health endpoint for convenience
    go func() {
        http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })
        http.ListenAndServe(":8086", nil)
    }()

    for {
        conn, err := ln.Accept(nil)
        if err != nil { continue }
        go handleConn(conn, upstream)
    }
}

func handleConn(conn quic.Connection, upstream string) {
    defer conn.CloseWithError(0, "bye")
    // Each stream is a bidirectional UDP channel: first frame: target JSON, then length-prefixed datagrams both ways
    for {
        st, err := conn.AcceptStream(nil)
        if err != nil { return }
        go func(s quic.Stream) {
            defer s.Close()
            // read target json (length-prefixed uint16)
            hdr := make([]byte, 2)
            if _, err := io.ReadFull(s, hdr); err != nil { return }
            nlen := binary.BigEndian.Uint16(hdr)
            buf := make([]byte, nlen)
            if _, err := io.ReadFull(s, buf); err != nil { return }
            var tgt udpTarget
            if err := json.Unmarshal(buf, &tgt); err != nil { return }
            if tgt.Addr == "" { tgt.Addr = upstream }
            // open UDP socket
            raddr, err := net.ResolveUDPAddr("udp", tgt.Addr)
            if err != nil { return }
            c, err := net.DialUDP("udp", nil, raddr)
            if err != nil { return }
            defer c.Close()
            // backflow: UDP -> QUIC (len-prefixed);
            go func() {
                rbuf := make([]byte, 64*1024)
                for {
                    n, _, err := c.ReadFromUDP(rbuf)
                    if n > 0 {
                        lb := make([]byte, 2)
                        binary.BigEndian.PutUint16(lb, uint16(n))
                        s.Write(lb)
                        s.Write(rbuf[:n])
                    }
                    if err != nil { return }
                }
            }()
            // forward: QUIC -> UDP (len-prefixed)
            for {
                if _, err := io.ReadFull(s, hdr); err != nil { return }
                nlen := binary.BigEndian.Uint16(hdr)
                if int(nlen) == 0 { continue }
                p := make([]byte, int(nlen))
                if _, err := io.ReadFull(s, p); err != nil { return }
                c.Write(p)
            }
        }(st)
    }
}

// Reuse insecure TLS generator from ingress/quic (copy minimal to keep file standalone)
import (
    "bytes"
    "crypto/rand"
    "crypto/rsa"
    "crypto/tls"
    "crypto/x509"
    "crypto/x509/pkix"
    "encoding/pem"
    "math/big"
)

func generateInsecureTLSConfig() *tls.Config {
    cert, key := generateSelfSigned()
    tlsCert, _ := tls.X509KeyPair(cert, key)
    return &tls.Config{Certificates: []tls.Certificate{tlsCert}, NextProtos: []string{"shieldx-masque"}}
}

func generateSelfSigned() (certPEM, keyPEM []byte) {
    priv, _ := rsa.GenerateKey(rand.Reader, 2048)
    tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "shieldx-masque"}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour)}
    der, _ := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
    certBuf := &bytes.Buffer{}
    pem.Encode(certBuf, &pem.Block{Type: "CERTIFICATE", Bytes: der})
    keyBuf := &bytes.Buffer{}
    pem.Encode(keyBuf, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
    return certBuf.Bytes(), keyBuf.Bytes()
}


