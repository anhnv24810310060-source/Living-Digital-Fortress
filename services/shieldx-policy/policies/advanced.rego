package shieldx.routing

default allow = false

# Allow authenticated requests to /api/
allow {
    input.path = "/api/v1/users"
    input.scope = "read:users"
}

# Allow health checks
allow {
    input.path = "/health"
}

# Allow metrics
allow {
    input.path = "/metrics"
}

# Deny suspicious paths
deny {
    contains(input.path, "..")
}

deny {
    contains(input.path, "etc/passwd")
}
