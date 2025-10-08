package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"time"
)

func getenvInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func main() {
	port := getenvInt("DECOY_SSH_PORT", 2222)
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	log.Printf("[decoy-ssh] listening on :%d", port)
	banner := os.Getenv("DECOY_SSH_BANNER")
	if banner == "" {
		banner = "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5"
	}
	for {
		c, err := ln.Accept()
		if err != nil {
			continue
		}
		go func(conn net.Conn) {
			defer conn.Close()
			conn.SetWriteDeadline(time.Now().Add(2 * time.Second))
			fmt.Fprintf(conn, "%s\r\n", banner)
			time.Sleep(2 * time.Second)
		}(c)
	}
}
