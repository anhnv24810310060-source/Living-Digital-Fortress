package shieldx.authz

deny[msg] {
  input.method == "POST"
  input.path[0] == "admin"
  msg := "admin POSTs are disabled in demo"
}
