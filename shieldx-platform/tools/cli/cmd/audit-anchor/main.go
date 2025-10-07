package main

import (
    "flag"
    "fmt"
    "log"
    "time"

    "shieldx/pkg/audit"
)

func main() {
    file := flag.String("file", "data/ledger-ingress.log", "ledger file to hash")
    flag.Parse()
    h, err := audit.HashChain(*file)
    if err != nil { log.Fatalf("hash: %v", err) }
    fmt.Printf("%s %s\n", time.Now().UTC().Format(time.RFC3339), h)
}



