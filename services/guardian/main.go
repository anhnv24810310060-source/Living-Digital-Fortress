package main

import (
    "encoding/binary"
    "encoding/json"
    "fmt"
    "log"
    "net"
    "net/http"
    "os"
    "strconv"
    "time"
    "io"
    "context"

    "crypto/ecdh"
    "crypto/tls"
    quic "github.com/quic-go/quic-go"
    "shieldx/pkg/wch"
    "shieldx/pkg/metrics"
    "strings"
)

func getenvInt(key string, def int) int {
    v := os.Getenv(key)
    if v == "" { return def }
    n, err := strconv.Atoi(v)
    if err != nil { return def }
    return n
}

func main() {
    // This is the real server protected behind ingress. It only listens on loopback.
    port := getenvInt("GUARDIAN_PORT", 9090)
    // Static X25519 key for guardian (PoC). In production rotate per process and per channel.
    curve := ecdh.X25519()
    guardianPriv, err := curve.GenerateKey(nil)
    if err != nil { log.Fatalf("guardian key: %v", err) }
    guardianPub := guardianPriv.PublicKey().Bytes()

    mux := http.NewServeMux()
    reg := metrics.NewRegistry()
    mRecv := metrics.NewCounter("guardian_wch_recv_total", "Total sealed envelopes received")
    mUDPRecv := metrics.NewCounter("guardian_wch_udp_recv_total", "Total sealed UDP envelopes received")
    mMasqueOK := metrics.NewCounter("guardian_masque_success_total", "MASQUE QUIC UDP relays succeeded")
    mMasqueFB := metrics.NewCounter("guardian_masque_fallback_total", "MASQUE QUIC fallback to direct UDP")
    mUDPDir := metrics.NewCounter("guardian_udp_direct_total", "Direct UDP relays succeeded")
    mUDPErr := metrics.NewCounter("guardian_udp_error_total", "UDP relay errors")
    mRekeyGrace := metrics.NewCounter("guardian_rekey_grace_hits_total", "Decrypt succeeded using counter-1 grace window")
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Guardian real service OK. Path=%s\n", r.URL.Path)
    })
    mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })
    // Publish guardian public key (base64)
    mux.HandleFunc("/wch/pubkey", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        _ = json.NewEncoder(w).Encode(map[string]string{"pubKey": wch.MarshalB64(guardianPub)})
    })
    // Receive sealed envelope, decrypt, route to local handler, encrypt response
    mux.HandleFunc("/wch/recv", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        // Limit envelope size to prevent memory abuse
        maxBytes := getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536)
        r.Body = http.MaxBytesReader(w, r.Body, int64(maxBytes))
        mRecv.Inc()
        var env wch.Envelope
        if err := json.NewDecoder(r.Body).Decode(&env); err != nil { http.Error(w, "bad request", 400); return }
        if env.ChannelID == "" || env.EphemeralPubB64 == "" || env.NonceB64 == "" || env.CiphertextB64 == "" {
            http.Error(w, "missing fields", 400)
            return
        }
        clientPubBytes, err := wch.UnmarshalB64(env.EphemeralPubB64)
        if err != nil { http.Error(w, "bad client key", 400); return }
        nonce, err := wch.UnmarshalB64(env.NonceB64)
        if err != nil { http.Error(w, "bad nonce", 400); return }
        ct, err := wch.UnmarshalB64(env.CiphertextB64)
        if err != nil { http.Error(w, "bad ciphertext", 400); return }
        clientPub, err := curve.NewPublicKey(clientPubBytes)
        if err != nil { http.Error(w, "bad client pub", 400); return }
        shared := guardianPriv.ECDH(clientPub)
        var key []byte
        var pt []byte
        // Try current counter and a grace window of counter-1 to avoid race during rekey
        tryCounters := []int{env.RekeyCounter}
        if env.RekeyCounter > 0 { tryCounters = append(tryCounters, env.RekeyCounter-1) }
        var decErr error
        for _, c := range tryCounters {
            if c > 0 {
                key, err = wch.DeriveKeyWithCounter(shared, env.ChannelID, c)
            } else {
                key, err = wch.DeriveKey(shared, env.ChannelID)
            }
            if err != nil { decErr = err; continue }
            pt, decErr = wch.Open(key, nonce, ct)
            if decErr == nil {
                if c == env.RekeyCounter-1 { mRekeyGrace.Inc() }
                break
            }
        }
        if decErr != nil { http.Error(w, "decrypt error", 400); return }
        // Unmarshal inner request and handle locally
        var inner wch.InnerRequest
        if err := json.Unmarshal(pt, &inner); err != nil { http.Error(w, "inner parse", 400); return }
        // For PoC, only GET / is supported
        status := 200
        body := []byte(fmt.Sprintf("Hello from Guardian over Whisper. Path=%s", inner.Path))
        resp := wch.InnerResponse{Status: status, Headers: map[string]string{"Content-Type":"text/plain"}, Body: body}
        respJSON := wch.ToJSON(resp)
        nonce2, ct2, err := wch.Seal(key, respJSON)
        if err != nil { http.Error(w, "encrypt error", 500); return }
        _ = json.NewEncoder(w).Encode(map[string]string{
            "nonce": wch.MarshalB64(nonce2),
            "ciphertext": wch.MarshalB64(ct2),
        })
    })
    // Receive sealed UDP request, relay via MASQUE QUIC and seal response
    mux.HandleFunc("/wch/recv-udp", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        // Limit envelope size to prevent memory abuse
        maxBytes := getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536)
        r.Body = http.MaxBytesReader(w, r.Body, int64(maxBytes))
        mUDPRecv.Inc()
        var env wch.Envelope
        if err := json.NewDecoder(r.Body).Decode(&env); err != nil { http.Error(w, "bad request", 400); return }
        clientPubBytes, err := wch.UnmarshalB64(env.EphemeralPubB64)
        if err != nil { http.Error(w, "bad client key", 400); return }
        nonce, err := wch.UnmarshalB64(env.NonceB64)
        if err != nil { http.Error(w, "bad nonce", 400); return }
        ct, err := wch.UnmarshalB64(env.CiphertextB64)
        if err != nil { http.Error(w, "bad ciphertext", 400); return }
        clientPub, err := curve.NewPublicKey(clientPubBytes)
        if err != nil { http.Error(w, "bad client pub", 400); return }
        shared := guardianPriv.ECDH(clientPub)
        var key []byte
        // Grace window for counter-1 when decrypting
        tryCounters := []int{env.RekeyCounter}
        if env.RekeyCounter > 0 { tryCounters = append(tryCounters, env.RekeyCounter-1) }
        var pt []byte
        var decErr error
        for _, c := range tryCounters {
            if c > 0 { key, err = wch.DeriveKeyWithCounter(shared, env.ChannelID, c) } else { key, err = wch.DeriveKey(shared, env.ChannelID) }
            if err != nil { decErr = err; continue }
            pt, decErr = wch.Open(key, nonce, ct)
            if decErr == nil {
                if c == env.RekeyCounter-1 { mRekeyGrace.Inc() }
                break
            }
        }
        if decErr != nil { http.Error(w, "decrypt error", 400); return }
        // Decrypt inner UDP request done in grace loop above; pt holds plaintext
        var udpReq wch.InnerUDPRequest
        if err := json.Unmarshal(pt, &udpReq); err != nil { http.Error(w, "inner parse", 400); return }
        // Relay via MASQUE QUIC if configured, else direct UDP
        var respBytes []byte
        if masque := os.Getenv("MASQUE_QUIC_ADDR"); masque != "" {
            if b, err := masqueSingleExchange(masque, udpReq.Target, udpReq.Data, udpReq.TimeoutMs); err == nil {
                respBytes = b
                mMasqueOK.Inc()
            } else {
                respBytes, _ = directUDPSingleExchange(udpReq.Target, udpReq.Data, udpReq.TimeoutMs)
                mMasqueFB.Inc()
            }
        } else {
            respBytes, _ = directUDPSingleExchange(udpReq.Target, udpReq.Data, udpReq.TimeoutMs)
            if len(respBytes) > 0 { mUDPDir.Inc() } else { mUDPErr.Inc() }
        }
        udpResp := wch.UDPResponse{Data: respBytes}
        respJSON := wch.ToJSON(udpResp)
        nonce2, ct2, err := wch.Seal(key, respJSON)
        if err != nil { http.Error(w, "encrypt error", 500); return }
        _ = json.NewEncoder(w).Encode(map[string]string{"nonce": wch.MarshalB64(nonce2), "ciphertext": wch.MarshalB64(ct2)})
    })
    reg.Register(mRecv); reg.Register(mUDPRecv); reg.Register(mMasqueOK); reg.Register(mMasqueFB); reg.Register(mUDPDir); reg.Register(mUDPErr); reg.Register(mRekeyGrace); mux.Handle("/metrics", reg)

    addr := fmt.Sprintf("127.0.0.1:%d", port)
    log.Printf("[guardian] listening on http://%s", addr)
    log.Fatal(http.ListenAndServe(addr, mux))
}


type masqueTarget struct { Addr string `json:"addr"` }

func directUDPSingleExchange(target string, payload []byte, timeoutMs int) ([]byte, error) {
    raddr, err := net.ResolveUDPAddr("udp", target)
    if err != nil { return nil, err }
    c, err := net.DialUDP("udp", nil, raddr)
    if err != nil { return nil, err }
    defer c.Close()
    _, _ = c.Write(payload)
    _ = c.SetReadDeadline(time.Now().Add(time.Duration(timeoutMs+500) * time.Millisecond))
    buf := make([]byte, 64*1024)
    n, _, err := c.ReadFromUDP(buf)
    if err != nil { return nil, err }
    return buf[:n], nil
}

func masqueSingleExchange(addrList, target string, payload []byte, timeoutMs int) ([]byte, error) {
    // Support multiple addresses (comma separated) with simple retry policy
    addrs := []string{}
    for _, a := range strings.Split(addrList, ",") { if s := strings.TrimSpace(a); s != "" { addrs = append(addrs, s) } }
    if len(addrs) == 0 { addrs = []string{addrList} }
    retries := getenvInt("MASQUE_RETRY", 2)
    tout := getenvInt("MASQUE_TIMEOUT_MS", timeoutMs+1000)
    if tout < 500 { tout = 500 }
    var lastErr error
    for attempt := 0; attempt <= retries; attempt++ {
        for _, addr := range addrs {
            tlsConf := &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"shieldx-masque"}}
            conn, err := quic.DialAddr(addr, tlsConf, &quic.Config{})
            if err != nil { lastErr = err; continue }
            func() {
                defer conn.CloseWithError(0, "bye")
                ctx, cancel := context.WithTimeout(context.Background(), time.Duration(tout)*time.Millisecond)
                defer cancel()
                st, err := conn.OpenStreamSync(ctx)
                if err != nil { lastErr = err; return }
                defer st.Close()
                // send target header (length-prefixed JSON)
                hdr := make([]byte, 2)
                tgt, _ := json.Marshal(masqueTarget{Addr: target})
                if len(tgt) > 65535 { lastErr = fmt.Errorf("target too long"); return }
                binary.BigEndian.PutUint16(hdr, uint16(len(tgt)))
                if _, err := st.Write(hdr); err != nil { lastErr = err; return }
                if _, err := st.Write(tgt); err != nil { lastErr = err; return }
                // send payload frame
                if len(payload) > 65535 { payload = payload[:65535] }
                binary.BigEndian.PutUint16(hdr, uint16(len(payload)))
                if _, err := st.Write(hdr); err != nil { lastErr = err; return }
                if _, err := st.Write(payload); err != nil { lastErr = err; return }
                // read one response frame
                if _, err := io.ReadFull(st, hdr); err != nil { lastErr = err; return }
                nlen := binary.BigEndian.Uint16(hdr)
                if nlen == 0 { lastErr = nil; return }
                buf := make([]byte, int(nlen))
                if _, err := io.ReadFull(st, buf); err != nil { lastErr = err; return }
                // success
                lastErr = nil
                payload = buf
            }()
            if lastErr == nil { return payload, nil }
        }
        time.Sleep(time.Duration(100*(attempt+1)) * time.Millisecond)
    }
    return nil, lastErr
}

