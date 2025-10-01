package shieldx.authz

default allow = false

# Allow health and metrics by default
allow {
  input.path == ["healthz"]
}

allow {
  input.path == ["metrics"]
}

# Example allowed GET on public info
allow {
  input.method == "GET"
  input.path[0] == "public"
}
