package hphook

import (
    decoyhttp "shieldx/services/honeypot-service/internal/decoys/http"
    "shieldx/shared/eventbus"
)

// StartHTTPDecoy starts the honeypot HTTP decoy on the given address with the provided bus.
func StartHTTPDecoy(addr string, bus *eventbus.Bus) error { return decoyhttp.New(addr, bus).Start() }
