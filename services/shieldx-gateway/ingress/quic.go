//go:build quic

package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"shieldx/shared/shieldx-common/pkg/metrics"
	"shieldx/shared/shieldx-common/pkg/wch"

	quic "github.com/quic-go/quic-go"
)

// startQUICServer launches a minimal QUIC server that accepts streams carrying wch.Envelope JSON
// and forwards them to Guardian's /wch/recv, returning the response bytes to the same stream.
var (
	mQUICPathChange = metrics.NewCounter("quic_path_change_total", "QUIC path change events")
	mQUICDgramRecv  = metrics.NewCounter("quic_datagram_recv_total", "QUIC datagrams received")
	mQUICDgramSent  = metrics.NewCounter("quic_datagram_sent_total", "QUIC datagrams sent")
	mQUICConnOpen   = metrics.NewCounter("quic_conn_open_total", "QUIC connections opened")
	mQUICConnClose  = metrics.NewCounter("quic_conn_close_total", "QUIC connections closed")
	gQUICConns      = metrics.NewGauge("quic_conns", "Current active QUIC connections")
	mQUICPingRecv   = metrics.NewCounter("quic_ping_recv_total", "QUIC ping datagrams received")
	mQUICPongSent   = metrics.NewCounter("quic_pong_sent_total", "QUIC pong datagrams sent")
)

func startQUICServer(addr string) error {
	tlsConf := generateRotatingTLSConfig()
	keepAlive := time.Duration(getenvInt("INGRESS_QUIC_KEEPALIVE_SEC", 15)) * time.Second
	idle := time.Duration(getenvInt("INGRESS_QUIC_IDLE_SEC", 45)) * time.Second
	qconf := &quic.Config{MaxIncomingStreams: 1024, EnableDatagrams: true, KeepAlivePeriod: keepAlive, MaxIdleTimeout: idle, Allow0RTT: true}
	ln, err := quic.ListenAddr(addr, tlsConf, qconf)
	if err != nil {
		return err
	}
	log.Printf("[ingress] QUIC listening on %s", addr)
	// register metrics (reg is shared from main package)
	if reg != nil {
		reg.Register(mQUICPathChange)
		reg.Register(mQUICDgramRecv)
		reg.Register(mQUICDgramSent)
		reg.Register(mQUICConnOpen)
		reg.Register(mQUICConnClose)
		reg.RegisterGauge(gQUICConns)
		reg.Register(mQUICPingRecv)
		reg.Register(mQUICPongSent)
	}
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for now := range ticker.C {
			chanReg.purgeExpired(now)
		}
	}()
	go func() {
		for {
			conn, err := ln.Accept(nil)
			if err != nil {
				continue
			}
			mQUICConnOpen.Inc()
			gQUICConns.Set(gQUICConns.Get() + 1)
			// path watcher works even when client only uses streams
			go watchQUICPath(conn)
			// datagram loop on this connection
			go handleQUICDatagrams(conn)
			// stream loop with close accounting
			go func(c quic.Connection) {
				defer func() {
					mQUICConnClose.Inc()
					if g := gQUICConns.Get(); g > 0 {
						gQUICConns.Set(g - 1)
					}
				}()
				handleQUICConn(c)
			}(conn)
		}
	}()
	return nil
}

func handleQUICConn(conn quic.Connection) {
	defer conn.CloseWithError(0, "bye")
	for {
		stream, err := conn.AcceptStream(nil)
		if err != nil {
			return
		}
		go func(st quic.Stream) {
			defer st.Close()
			// read all JSON
			b, err := io.ReadAll(&io.LimitedReader{R: st, N: int64(getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536))})
			if err != nil {
				return
			}
			var env wch.Envelope
			if err := json.Unmarshal(b, &env); err != nil {
				return
			}
			info, ok := chanReg.get(env.ChannelID)
			if !ok || time.Now().After(info.Expiry) {
				return
			}
			// forward to guardian
			guardian := fmt.Sprintf("http://127.0.0.1:%d/wch/recv", getenvInt("GUARDIAN_PORT", 9090))
			resp, err := httpPost(guardian, b)
			if err != nil {
				return
			}
			st.Write(resp)
		}(stream)
	}
}

func handleQUICDatagrams(conn quic.Connection) {
	lastAddr := remoteAddrString(conn)
	for {
		b, err := conn.ReceiveMessage()
		if err != nil {
			return
		}
		mQUICDgramRecv.Inc()
		cur := remoteAddrString(conn)
		if cur != lastAddr {
			log.Printf("[ingress] QUIC path changed: %s -> %s", lastAddr, cur)
			mQUICPathChange.Inc()
			lastAddr = cur
		}
		var env wch.Envelope
		if len(b) == 0 {
			continue
		}
		// Handle ping pong for liveness/migration keepalive
		if isPingDatagram(b) {
			mQUICPingRecv.Inc()
			if err := conn.SendMessage([]byte(`{"pong":true}`)); err == nil {
				mQUICPongSent.Inc()
			}
			continue
		}
		if err := json.Unmarshal(b, &env); err != nil {
			continue
		}
		info, ok := chanReg.get(env.ChannelID)
		if !ok || time.Now().After(info.Expiry) {
			continue
		}
		guardian := fmt.Sprintf("http://127.0.0.1:%d/wch/recv", getenvInt("GUARDIAN_PORT", 9090))
		resp, err := httpPost(guardian, b)
		if err != nil {
			continue
		}
		if err := conn.SendMessage(resp); err == nil {
			mQUICDgramSent.Inc()
		}
	}
}

// watchQUICPath observes remote address changes even if the client only uses streams.
func watchQUICPath(conn quic.Connection) {
	prev := remoteAddrString(conn)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		cur := remoteAddrString(conn)
		if cur == "" {
			return
		}
		if cur != prev {
			log.Printf("[ingress] QUIC path changed (watch): %s -> %s", prev, cur)
			mQUICPathChange.Inc()
			prev = cur
		}
	}
}

func remoteAddrString(conn quic.Connection) string {
	ra := conn.RemoteAddr()
	if ra == nil {
		return ""
	}
	// Normalize to host:port
	if ua, ok := ra.(*net.UDPAddr); ok {
		return ua.String()
	}
	return ra.String()
}

func isPingDatagram(b []byte) bool {
	// minimal check to avoid full JSON parse cost on hot path
	if len(b) > 64 {
		return false
	}
	// expect {"ping":true}
	return bytes.Contains(b, []byte("\"ping\""))
}

// httpPost is a tiny helper to post JSON bytes and read response body
func httpPost(url string, body []byte) ([]byte, error) {
	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// generateInsecureTLSConfig returns a minimal TLS config for QUIC (self-signed). In production, use real certs.
var currentCert atomic.Value // *tls.Certificate

func generateRotatingTLSConfig() *tls.Config {
	// initial
	cert, key := generateSelfSigned()
	tlsCert, _ := tls.X509KeyPair(cert, key)
	currentCert.Store(&tlsCert)
	// rotate periodically
	rotateMin := getenvInt("QUIC_CERT_ROTATE_MIN", 10)
	go func() {
		t := time.NewTicker(time.Duration(rotateMin) * time.Minute)
		defer t.Stop()
		for range t.C {
			cert, key := generateSelfSigned()
			if tlsCert, err := tls.X509KeyPair(cert, key); err == nil {
				currentCert.Store(&tlsCert)
			}
		}
	}()
	return &tls.Config{
		GetCertificate: func(chi *tls.ClientHelloInfo) (*tls.Certificate, error) {
			c := currentCert.Load().(*tls.Certificate)
			return c, nil
		},
		NextProtos: []string{"shieldx-wch"},
	}
}

// The following helpers keep the example concise. Full implementations would manage certs securely.
func generateSelfSigned() (certPEM, keyPEM []byte) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "shieldx-ingress"}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour)}
	der, _ := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	certBuf := &bytes.Buffer{}
	pem.Encode(certBuf, &pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyBuf := &bytes.Buffer{}
	pem.Encode(keyBuf, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	return certBuf.Bytes(), keyBuf.Bytes()
}
