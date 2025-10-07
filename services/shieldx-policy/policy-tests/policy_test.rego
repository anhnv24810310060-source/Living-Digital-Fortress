package policytests

test_deny_when_run_as_root {
  input := data.fixtures.deployment
  some msg
  deny[msg] with input as input
}
