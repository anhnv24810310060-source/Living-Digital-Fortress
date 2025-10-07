package shieldx.authz

import data.shieldx.authz.allow
import data.shieldx.authz.deny

test_allow_healthz {
  allow with input as {"method":"GET","path":["healthz"]}
}

test_allow_public_get {
  allow with input as {"method":"GET","path":["public","info"]}
}

test_deny_admin_post {
  some msg
  deny[msg] with input as {"method":"POST","path":["admin","user"]}
}
