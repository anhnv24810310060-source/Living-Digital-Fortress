package main

import (
    "bufio"
    "fmt"
    "log"
    "net"
    "os"
    "strconv"
)

func getenvInt(key string, def int) int { v := os.Getenv(key); if v == "" { return def }; n, err := strconv.Atoi(v); if err != nil { return def }; return n }

func main() {
    port := getenvInt("DECOY_REDIS_PORT", 6380)
    ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
    if err != nil { log.Fatalf("listen: %v", err) }
    log.Printf("[decoy-redis] listening on :%d", port)
    banner := os.Getenv("DECOY_REDIS_BANNER")
    if banner == "" { banner = "+PONG\r\n" }
    for {
        c, err := ln.Accept()
        if err != nil { continue }
        go func(conn net.Conn){
            defer conn.Close()
            r := bufio.NewReader(conn)
            // read and ignore one line, respond with banner
            _, _ = r.ReadString('\n')
            fmt.Fprintf(conn, "%s", banner)
        }(c)
    }
}



