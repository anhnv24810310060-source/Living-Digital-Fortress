package shieldx.rbac

default allow = false

# Example RBAC policy bridging claims to resources/actions
allow {
  input.roles[_] == "admin"
}

allow {
  input.roles[_] == "user"
  input.resource == "api"
  input.action == "read"
}

allow {
  input.roles[_] == "service"
  input.resource == "metrics"
  input.action == "write"
}
