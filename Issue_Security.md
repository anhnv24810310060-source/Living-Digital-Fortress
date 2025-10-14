# Security Findings and Improvement Proposals

Day: 2025-10-13

This document summarizes security vulnerabilities and points for improvement for each service/component. Each item includes: (1) sequence number, (2) defect description, (3) proposed solution and location, (4) completion criteria.

---

### A. General configuration / Infrastructure (docker-compose, images, secrets)

#### 1) Exposed/weak secrets in docker-compose (dev defaults) - Issue #41
- Description: Secrets/creds are in docker-compose and have weak/default values:
- POSTGRES_PASSWORD=shieldx123; JWT_SECRET=dev-jwt-secret; ADMISSION_SECRET=dev-secret-12345; Grafana admin password fortress123; ML_API_ADMIN_TOKEN="" (empty) → disable admin protection.
- Location: `docker-compose.full.yml` (postgres, auth-service, ingress, grafana, ml-orchestrator).
- Recommendations:
- Move secrets to Docker/K8s secrets; remove default values ​​in compose production.
- Force strong configuration via environment variable/secret file when deploying; enable periodic rotation.
- Separate dev and prod profiles; in prod, prohibit committing secrets and prohibit weak defaults.
- Done when: Secrets are no longer present in the repo; pipelines/helm charts require secret inputs; runtime checks confirm default variables are not used in prod.



#### 2) Redis/Postgres without hardening - Issue #42
- Description: Redis without auth; Postgres uses weak passwords and default port/volume; risk of lateral movement if internal network is exposed.
- Location: `docker-compose.full.yml` (redis, postgres).
- Recommendation: Enable auth/password for Redis; use separate network segments; Postgres uses strong user/pass, TLS between app↔DB, enable pgaudit in prod.
- Complete when: Redis/PG auth configuration applied; check connection requires cred; security baseline doc updated.


#### 3) Pin container version Images - Issue #43
- Description: Prometheus/Grafana uses `latest`; SBOM/CVE control is difficult.
- Location: `docker-compose.full.yml` (prometheus, grafana).
- Recommendation: Pin specific version or digest; add image scanning (Trivy) in CI.
- Done when: All images pin version/digest; CI reports scan pass.

---

### B. Ingress Gateway

#### 4) mTLS is not required in current prod configuration - Issue #44
- Description: `RATLS_REQUIRE_CLIENT_CERT=false` in compose dev; Admission header is the main protection mechanism → weak if secret is exposed.
- Location: `services/shieldx-gateway/ingress/main.go`, `docker-compose.full.yml` (ingress env).
- Suggestion: Prod enables RA-TLS + requires client cert; configure allowlist SAN via `INGRESS_ALLOWED_CLIENT_SAN_PREFIXES`; save Admission secret in secret store and rotate.
- Complete when: Ingress requires mTLS in prod; test handshake client cert pass/fail; secret not in repo.



#### 5) Admission HMAC with 1 minute replay window - Issue #45
- Description: Token valid in current/previous minute bucket → can be replayed in ~60s if sniffed.
- Location: `pkg/guard/guard.go` VerifyHeader.
- Suggestion: Narrow the window (e.g. 30s), add one-time nonce (LRU cache), and IP/source binding. HTTPS/mTLS required.
- Done when: Unit test VerifyHeader updated; metrics for reject replay; pen test not replayable.


#### 6) Expose /metrics and /health public - Issue #46
- Description: /metrics, /health are usually open; can leak internal information.
- Location: Ingress and services in general.
- Recommendation: Only expose internally (network policy) or protect with auth/gateway; hide sensitive information, add rate-limit.
- Done when: Network policy applies; external scans do not access /metrics.
---

### C. Policy Rollout Service


#### 7) Unauthenticated policy apply + SSRF possible - Issue #47
- Description: `/apply` gets `url` and loads bundle via `http.Get`, optionally `sigURL`. No control over schema/host/redirect/size → SSRF/DoS.
- Location: `services/shieldx-policy/policy-rollout/main.go` (function `fetchAndVerify`).
- Recommendation:
- Force auth (JWT/mTLS/Admission) + RBAC for `/apply`.
- URL Validator: allow only `https`, block private IP (127.0.0.0/8, 10/8, 172.16/12, 192.168/16, link-local, 169.254/16), block file://, gopher://, ftp://, and block redirects.
- Set short timeout, size limit (Content-Length and io.LimitReader), check Content-Type/ZIP, verify cert.
- Enforce signature (remove `NoopVerifier` in prod), key management process for cosign.
- Done when: Add middleware auth; unit/integration test SSRF passes; large load benchmark is blocked; logs/metrics reflect reject case.


#### 8) No size/time limits when downloading bundle - Issue #48
- Description: Read all into RAM (bytes → zip reader) → easy to DoS.
- Location: `policy-rollout/main.go` (`fetchAndVerify`).
- Recommendation: Use http.Client with timeout; io.LimitReader (e.g. 20MB); reject if exceeding threshold.
- Complete when: Try downloading file > threshold is 413/400; memory stable in test.
---

### D. ThreatGraph
#### 9) ThreatGraph Writer: Arbitrary Cypher + No Auth - Issue  #49
- Description: `/graph/query` takes Cypher string directly from query param and executes; `/graph/node`/`/graph/edge` writes data without auth.
- Location: `services/shieldx-forensics/threatgraph/writer.go`.
- Recommendations:
- Mandatory auth (JWT/mTLS) and RBAC; remove arbitrary query endpoint.
- Allow only parameterized queries; add rate limit + audit log.
- Complete when: Test cannot run arbitrary Cypher; mandatory authentication testing.


#### 10) ThreatGraph API missing auth/rate-limit - Issue #50
- Description: `/ingest`, `/query`, `/stats` do not have auth/rate-limit.
- Location: `services/shieldx-forensics/threatgraph/main.go`.
- Recommendation: Add middleware auth (JWT/mTLS) + rate limit and input validation; schema for `ThreatQueryRequest`.
- Complete when: E2E test requires token; rate-limit works.
---

## E. ML Orchestrator

#### 11) Admin endpoints are not protected when token is empty - Issue #51
- Description: `makeAdminMiddleware()` returns passthrough if `ML_API_ADMIN_TOKEN==""`; in compose dev leave it empty → everyone can call /train, /model/*.
- Location: `services/shieldx-ml/ml-orchestrator/main.go` and `docker-compose.full.yml`.
- Recommendation: Enforce non-empty token in all environments; support mTLS or Admission guard for admin APIs; IP restriction.
- Complete when: Call without token returns 401; wrong token 401; correct token 2xx; test CI cover.


#### 12) Upload/model ops missing quota - Issue #52
- Description: Potential DoS if uploading large or multipart files; (code handles this but needs to confirm quotas/limits).
- Location: `ml-orchestrator/main.go` (upload/model save/load handler).
- Suggestion: Set `MaxBytesReader`, limit on file count/size, timeout; check format.
- Complete when: Upload exceeds threshold returns 413; log/metrics reflect.
---

## F. Camouflage API (Deception)

#### 13)CORS "*" and weak default API key - Issue #53
- Description: `Access-Control-Allow-Origin: *`. `validateToken` function uses `CAMOUFLAGE_API_KEY` with default "default_key"; if not set correctly → weak/known token.
- Location: `services/shieldx-deception/camouflage-api/main.go`.
- Recommendation: Restrict CORS origin (allowlist); force strong API key from secrets; consider JWT/mTLS; log and rate-limit.
- Done when: CORS only allows valid origins; require Authorization with valid key.
---

### G. Admin WebAPI

#### 14) Proxy endpoints without auth - Issue #54
- Description: `/api/shadow/*` proxy to Shadow without authentication.
- Location: `services/shieldx-admin/webapi/main.go`.
- Recommendations: Mandatory JWT/mTLS; role-based access; rate-limit and audit log.
- Complete when: No token → 401; wrong token → 401; correct token → 2xx; test OK.
---

### H. Verifier Pool

#### 15) `/nodes` public, access control required - Issue  #55
- Description: Purpose of node listing; comment "requires access control" but currently returns non-auth information.
- Location: `services/shieldx-auth/verifier-pool/main.go` (handleListNodes).
- Recommendations: JWT/mTLS protection; hide sensitive information; limit fields; rate-limit.
- Complete when: `/nodes` requires auth; contract test updated.
---

### I. Deception DSL Loader

#### 16) LoadFromURL lacks source control - Issue  #56
- Description: `LoadFromURL` calls `http.Get(url)` directly; if URL is provided by user → SSRF/DoS similar (redirect/size/timeouts).
- Location: `shared/shieldx-common/pkg/deception/dsl.go`.
- Recommendation: Allow `https` only, timeout and size limit, prohibit local addresses, disable redirects; consider loading via proxy allowlist.
- Done when: Unit test SSRF passes; limit applies.
---

### J. TLS/Crypto Hygiene

#### 17)HTTP (dev) service not yet TLS secured in prod - Issue #57
- Description: Many services listen to plain HTTP in dev.
- Location: many `main.go`; compose env.
- Recommendation: Enable RA-TLS/mTLS in prod; enforce TLS 1.3; prohibit HTTP except internal health; standard security header (HSTS when TLS is present).
- Done when: Prod only accepts TLS; check sslyze/nmap pass.

- 
#### 18)Key management/rotation - Issue #58
- Description: No rotation keys process (JWT, Admission, Cosign keyref) found.
- Location: public; `go.mod`/docs not recorded.
- Recommendation: Design rotation process; use secret manager; TTL/rotation policy; logging rotation events.
- Complete when: SOP/Docs and rotation script available; audit log confirmed.
---

### K. Observability và Data Exposure

#### 19) /metrics can expose information (paths, durations) - Issue  #59
- Description: Metrics data can leak internal paths, service names.
- Location: all services.
- Recommendation: Only scrape from internal Prometheus; filter sensitive labels; do not expose publicly.
- Complete when: Prometheus target only listens privately; external testing is not accessible.

- 
#### 20) Logs contain potentially sensitive data - Issue #60
- Description: Some handlers reflect headers/
- Location: ingress, gateway, deception…
- Recommendation: Redact PII/secrets in logs; add structured logging with policy.
- Done when: Check logs do not contain secrets; CI lint rule for log redaction.
- 
---

### L. Process & Defense in Depth

#### 21) Add rate limiting/circuit breaker for sensitive endpoints - Issue  #61
- Description: Some admin endpoints do not have limiter yet.
- Location: policy-rollout /apply, threatgraph writer, admin webapi.
- Suggestion: Middleware limiter by IP/user; CAPTCHA/2FA for admin GUI; circuit breaker with upstream.
- Complete when: Stress test does not crash the service; metrics limiter appears.

####  22) Add SAST/DAST in CI - Issue #62
- Description: No security pipelines found yet.
- Location: CI/CD.
- Suggestion: Gosec, Semgrep, Trivy image scan, dependency audit; PR policy block on fail.
- Complete when: CI pipelines add job security and quality gate enabled.

---

## Appendix: Priority Level (High → Low)
- High: (7), (9), (11), (1), (4), (8), (16)
- Medium: (5), (12), (14), (15), (6), (3), (2)
- Low/Improved: (17), (18), (19), (20), (21), (22)

---

Note: Some configurations are acceptable in dev environments (e.g. weak secrets, HTTP instead of TLS), but require strong measures in prod. The above points focus on addressing real exploitable vulnerabilities first.
---



 ## Added new discovery

#### 23) ORCH_ALLOW_INSECURE allows Orchestrator to run on HTTP - Issue #63
- Description: Variable `ORCH_ALLOW_INSECURE=1` enables pure HTTP mode (no TLS) for orchestrator, suitable for dev but if enabled by mistake in prod → MITM risk.
- Location: `services/shieldx-gateway/orchestrator/main.go` (block ORCH_ALLOW_INSECURE warning).
- Suggestion: Separate dev/prod profile; in prod completely remove this path or fail-fast when there is no TLS; add CI/CD guard to prevent setting this variable in prod.
- Complete when: Deploy prod without this flag; check endpoint to force TLS.



#### 24) MASQUE QUIC server uses self-signed insecure config - Issue #64
- Description: `generateInsecureTLSConfig()` function generates self-signed cert for QUIC MASQUE; acceptable for lab but not safe for prod.
- Location: `services/shieldx-gateway/masque/main.go`.
- Recommendation: Replace with RA-TLS or valid TLS from PKI; force client authentication.
- Complete when: QUIC server requests mTLS; test client authentication success/failure.

#### 25) Deception DSL LoadFromURL lacks redirect/size limit - Issue #65
- Description: `shared/shieldx-common/pkg/deception/dsl.go` and `pkg/deception/dsl.go` use `http.Get` and read entire body; lack timeout/limit.
- Location: corresponding `LoadFromURL` functions.
- Recommendation: Use http.Client with timeout; block redirects; io.LimitReader (e.g. 5–10MB); allow only https + allowlist hosts.

- Done when: Unit tests SSRF/DoS pass; limit size applies.

#### 26) ML Orchestrator upload/model operations need size constraints - Issue #66
- Description: Potentially large file uploads cause memory spikes if `MaxBytesReader` is missing on request.
- Location: `services/shieldx-ml/ml-orchestrator/main.go` (handlers /model/*, /train).
- Recommendation: Apply `r.Body = http.MaxBytesReader(w, r.Body, MAX)`; check Content-Length header; reject over threshold; validate MIME.
- Complete when: Load test > MAX returns 413.

#### 27) Admin WebAPI proxy without auth - Issue #67
- Description: `/api/shadow/*` does not control access; can abuse calls to Shadow service.
- Location: `services/shieldx-admin/webapi/main.go`.
- Recommendation: JWT/mTLS + RBAC; audit log; limit admin source IP.
- Complete when: No wrong token/role → 401/403; test passes.


#### 28) CORS “*” in Camouflage API - Issue #68
- Description: `Access-Control-Allow-Origin: *` and default API key `default_key` expose cross-origin.
Location: `services/shieldx-deception/camouflage-api/main.go`.
Recommendation: Allowlist domain; force strong key/JWT; default type in prod.
- Done when: CORS only accepts valid origins; key required.



#### 29) Verifier-pool `/nodes` exposes operational information - Issue #69
- Description: Returns node information without authentication; has a note “requires access control”.
- Location: `services/shieldx-auth/verifier-pool/main.go`.
- Recommendation: Protect JWT/mTLS; hide sensitive details; add rate-limit.
- Done when: `/nodes` requires auth; test valid.


#### 30) /metrics exposure requires network policy - Issue #70
- Description: Despite HELP/TYPE standardization, /metrics should still not be public.
Location: all services.
- Recommendation: Scrape only over internal network; prohibit public ports; add auth if forced to open.
- Done when: External scans do not access /metrics.
  
#### 31) Gateway JWT Secret has default dev-only-secret - Issue  #71
- Description: When `GATEWAY_JWT_SECRET` is not set, the service uses default `dev-only-secret` → easy to guess if running prod.
- Location: `services/shieldx-gateway/main.go`.
- Recommendation: Enforce secret from secret manager; fail-fast if variable does not exist in prod; support rotation.
- Complete when: Prod no longer has default secret; test env missing causes startup failure.

#### 32) Verifier-pool /validate missing auth - Issue #72 
- Description: Endpoint `/validate` handles verification requests without binding caller identity.
- Location: `services/shieldx-auth/verifier-pool/main.go`.
- Recommendation: JWT/mTLS protection; rate-limit; audit log.
- Done when: Call without token returns 401; token without authority returns 403.

#### 33) Orchestrator admin endpoints do not have their own guard - Issue #73
- Description: `/admin/pools` (and variants) require strong auth; currently only the general Admission guard is available.
- Location: `services/shieldx-gateway/orchestrator/main.go`.
- Recommendation: Add admin role request middleware (JWT/mTLS) for `/admin/*`; tighter rate-limit; audit log.
- Done when: Contract test admin requires separate permissions.

#### 34) Missing body size caps on some POST endpoints - Issue #74
- Description: Some services already have MaxBytesReader (contauth), but other endpoints (policy-rollout /apply, threatgraph writer) do not have size limits.
- Location: `services/shieldx-policy/policy-rollout/main.go`, `services/shieldx-forensics/threatgraph/writer.go`.
- Recommended: Apply `http.MaxBytesReader` (e.g. 10–20MB for bundle, <1–5MB for ingests) + Content-Length check + MIME.
- Complete when: Load > limit returns 413 and is logged.

#### 35) Path traversal/zip slip when unzipping zip - Issue #75
- Description: Policy bundle zip reads through `zip.NewReader` and loads content; need to validate internal paths when unzipping and reject `..`/absolute paths.
- Location: `services/shieldx-policy/policy-rollout/main.go` (load zip), `shared/shieldx-common/pkg/policy` (if unzipping file).
- Recommendation: When extracting files, normalize the path (filepath.Clean) and deny the path from the directory; scan entries before processing.
- Complete when: Unit test zip slip fail-case; linter rule added.

#### 36) OPA configuration allows /metrics & /health public - Issue #76
- Description: The demo rego rules allow /metrics and /health; if applied to prod without blocking external ingress, information will be exposed.
- Location: `services/shieldx-policy/policies/advanced.rego`, `.../demo/rules/allow.rego`.
- Recommendation: In prod profile, only allow internal or after auth; separate dev/prod rules set.
- Complete when: Prod rules do not allow public /metrics; check external access denied.

#### 37) CORS & Headers hardening is not consistent - Issue #77
- Description: Only Camouflage API has clear CORS; other services do not set security headers (HSTS, X-Content-Type-Options, CSP).
- Location: many services `main.go`.
- Recommendation: Add common middleware for security headers; enable HSTS in TLS; customize CSP in UI/API.
- Done when: Check headers via scan; standardize all services.

#### 38) QUIC/XDP extension path needs permission control - Issue #78
- Description: Ingress has XDP attach/QUIC server optional; need to ensure only enable with proper configuration and root rights; avoid unnecessary surfaces.
- Location: `services/shieldx-gateway/ingress/main.go` (XDP, QUIC), `.../masque/main.go`.
- Recommendation: Strict gating flag; require mTLS; audit log when enabled; default off prod if not required.
- Done when: Prod does not expose QUIC/XDP unless explicit; check port scan

#### 39) InsecureSkipVerify=true in Guardian QUIC client (MITM risk) - Issue #79
- Description: Guardian calls MASQUE via QUIC with `&tls.Config{InsecureSkipVerify: true}` → does not validate server certificate, may be MITM/impersonation.

- Location: `services/shieldx-forensics/guardian/main.go:683` (function `masqueSingleExchange`).

- Recommendation: Enable server authentication: provide trusted CA or certificate pinning (Public Key Pinning/SPKI); configure SAN/hostname check; consider mTLS.

- Complete when: QUIC connection fails if cert is wrong; integration test checks hostname/cert pin pass.

#### 40) Many services run pure HTTP (no TLS) at runtime - Issue #80
- Description: Services listen using `http.ListenAndServe` directly → if exposed to the network/k8s ingress can be sniffed/MITM.
- Location (typical example): 
- `services/shieldx-auth/auth-service/main.go:166` 
- `services/shieldx-policy/policy-rollout/main.go:199` 
- `services/shieldx-forensics/threatgraph/writer.go:240`, `.../threatgraph/main.go:334`, `.../anchor/main.go:74`, `.../guardian/main.go:637` 
- `services/shieldx-deception/decoy-manager/main.go:255`, `.../decoy-http/main.go:50`, `.../sinkhole/main.go:73`, `.../shapeshifter/main.go:294` 
- `services/shieldx-sandbox/autoheal/main.go:25`
- `services/shieldx-admin/marketplace/main.go:49`, `.../webapi/main.go:184`
- `shared/shieldx-sdk/plugin_registry/main.go:53`
- `services/shieldx-gateway/masque/main.go:64`
- Suggestion: Hide behind internal gateway/reuse standard TLS listener; or constrain networkPolicy to internal access only; enable mTLS when inter-service.

- Done when: Prod has no public HTTP port; nmap/sslyze testing confirms TLS/mTLS only.

#### 41) OAuth2 ClientSecret hardcoded in Auth Service (demo secrets) - Issue #81
- Description: `ClientSecret: "demo-secret-change-in-production"` and `"mobile-secret-change-in-production"` exist in code → vulnerable to abuse if deployed wrongly in prod.

- Location: `services/shieldx-auth/auth-service/main.go:87`, `:98`.

- Recommendation: Read client secret from secret store/environment variable; do not commit sample secret; fail-fast if secret is missing in prod profile.

- Complete when: Secrets are no longer in code; CI checks block when detecting this string; runtime requires valid secret.

  
#### 42) PQC Service: CORS "*" and weak default API key - Issue #82
- Description: `Access-Control-Allow-Origin: *` and `PQC_API_KEY` default `default_pqc_key` → vulnerable to abuse from arbitrary web origins.
- Location: `services/shieldx-auth/pqc-service/main.go:350` (CORS), `.../pqc-service/main.go` (function `validateToken`).
- Recommendation: Use allowlist origin; force strong API key from secret; consider JWT/mTLS instead of API key; add rate-limit/log.
- Done when: Only valid origins allowed; default key removed; CORS/401 test pass.

#### 43) Use crypto/md5 in Credits sharding (not suitable for security purposes) - Issue #83
- Description: Import `crypto/md5` in sharding mechanism; if used for security purposes (e.g. signing, tokenization) is not secure; if only hashing distributed keys is acceptable but better algorithms should be considered.
- Location: `services/shieldx-credits/credits/sharding_engine.go:5`.
- Recommendation: Confirm scope of use only for sharding (not secure); consider replacing with `sha256`/`xxhash` depending on requirements; clearly note in code.
- Complete when: Audit code confirms not using md5 for security; update docs/comment; (if changed) benchmark is ok.

#### 44) Command Injection in Gateway Ingress (using `sh -c` + format string) - Issue #84
- Description: Many shell commands built via `fmt.Sprintf` and executed using `exec.Command("sh", "-c", ...)` with runtime data (`iface`, `cidr`, `classid`, `altIface`, …) → injection risk if these parameters can be affected (env/inputs).
- Location: `services/shieldx-gateway/ingress/main.go` at lines: 154, 164-165, 728, 908, 1339-1363 (tc/ip/nft/wg commands).
- Recommendations:
- Whitelist values ​​(strict regex for interface name/IP/classid);
- Avoid `sh -c`, use `exec.Command` with separate parameters;

- Run in a low-privilege context (minimal capabilities); audit log commands;

- If shell is required, strongly escape and add unit test injection.

- Complete when: Command uses separate args; inputs are validated; test cases containing special characters are rejected.

#### 45) Dynamic SQL using `fmt.Sprintf` for table/column names in Credits - Issue #85
- Description: Query template is concatenated `fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)...")`; although values ​​use `$1..$n`, concatenating table/column names from input risks injection if not whitelisted.

- Location: `services/shieldx-credits/credits/multi_cloud_dr.go:367, 397, 412`.
- Recommendation: Whitelist valid tables/columns; do not accept from users; if configured, validate strictly; keep values ​​parameterized.
- Done when: Validation exists; tests with bad table names are rejected.

 #### 46) Ignore JSON unmarshal errors in Deception Shadow - Issue #86
- Description: Some `json.Unmarshal`s don't check for errors → invalid data can cause incorrect logic/hide errors.
- Location: `services/shieldx-deception/shadow/evaluator.go:553-554, 633`.
- Recommendation: Check for errors and return 400/log; add fuzz test for inputs.
- Done when: All Unmarshals handle errors; tests pass.

#### 47) Read the entire body without limits in some places (potential DoS) - Issue #87
- Description: Use `io.ReadAll` and ignore errors/no size limits.
- Example locations: `services/shieldx-deception/decoy-http/main.go:76`, `services/shieldx-ml/ml-orchestrator/main.go:823, 914`, `services/shieldx-gateway/ingress/main.go:276`.
- Recommendation: Use `http.MaxBytesReader` for requests; `io.LimitedReader` with reasonable threshold; enforce Content-Length and timeout.
- Done when: >threshold returns 413; benchmark memory stability.

#### 48) InsecureSkipVerify in tooling/tests (FYI) - Issue #88
- Description: Appears in CLI/test/README (not prod), but need to ensure it doesn't spread to runtime.

- Location: `tools/cli/cmd/wch-client/main.go:142,162,378`, `infrastructure/kubernetes/pilot/tests/red-team-test.go:49`, `pkg/wch/README.md`, `shared/shieldx-common/pkg/wch/README.md`, `pkg/ratls/ratls_test.go`.

- Recommendation: Explicitly annotate for test/dev only; check CI to prevent this flag in prod code.

- Done when: CI rule exists; no more InsecureSkipVerify in prod service code.

#### 49) REDIS_PASSWORD can be empty (weak auth) in Auth Service - Issue #89
- Description: Allow `REDIS_PASSWORD` to be empty → if Redis is exposed to the internal network, it can be accessed illegally.
- Location: `services/shieldx-auth/auth-service/main.go:20, 32, 51, 62`.
- Recommendation: Prod requires password/ACL; fail-fast if the variable is empty; TLS between app↔Redis if over the network.
- Complete when: Prod startup fails if missing password; healthcheck confirms ACL.

##### 50) MASQUE QUIC: TLS self-signed configuration without trust (dev path) - Issue #90 
- Description: `generateInsecureTLSConfig()` generates a self-signed cert for QUIC MASQUE (dev service), if enabled by mistake, prod will lack trust/security.
- Location: `services/shieldx-gateway/masque/main.go:147-150`.
- Suggestion: Split dev/prod profile; prod uses RA-TLS/valid cert; add client auth.
- Done when: Prod no longer has insecure codepath; mTLS test passes.

#### 51) JWT does not check Issuer/Audience binding (omits context validation) - Issue #91
- Description: `ValidateToken` function only checks signing method and expiry date; does not bind Issuer/Audience to expected value → can accept tokens valid for signature but issued from other issuers.
- Location: `pkg/auth/jwt_manager.go` and `shared/shieldx-common/pkg/auth/jwt_manager.go` (function `ValidateToken`).
- Suggestion: Use parser with strict option: `jwt.NewParser(jwt.WithValidMethods(["RS256"]), jwt.WithIssuer(expectedIssuer), jwt.WithAudience(expectedAud))`; after parsing, check `claims.Issuer`, `claims.Audience`, `claims.Subject` according to config; consider `kid` rotation and JWKS.

- Complete when: Unit test wrong token issuer/audience is rejected; issuer/audience configuration exists via environment variable/secret.

#### 52) ThreatGraph uses default Neo4j password "password" - Issue #92
- Description: Variable `NEO4J_PASSWORD` has default "password" → if running dev this configuration in an open network environment will be very weak.

- Location: `services/shieldx-forensics/threatgraph/writer.go` (`main()` block), with `neo4jPassword := getEnv("NEO4J_PASSWORD", "password")`.

- Recommendation: No default in prod; read from secret store; fail-fast if missing; enable Neo4j TLS and user with minimal privileges; change default password on initialization.

- Done when: Prod does not accept default; startup test fails if password is missing; connection requires TLS and strong cred.
---
==============================================================
### M. Additional Critical Findings (2025-10-13)

#### 53) Firecracker sandbox executes payload on host shell - Issue #93
- Description: `executeInVM()` in both `shared/shieldx-common/pkg/sandbox/firecracker_runner.go` and `pkg/sandbox/firecracker_runner.go` writes the payload to a temp shell script and runs `/bin/sh` on the host, never launching the MicroVM. Any caller can achieve arbitrary command execution on the host under sandbox credentials.
- Location: `FirecrackerRunner.executeInVM` helper in the files noted above.
- Recommendation: Ensure payloads execute inside the Firecracker guest (e.g. copy script into VM via API or use serial console). Refuse to run when VM startup fails, drop any direct host shell execution, and add integration tests that verify commands do not run if the VM layer is unavailable.
- Done when: Payloads run exclusively inside the VM, host execution paths are removed, and regression tests cover VM isolation failures.

#### 54) Auth Service login accepts any credentials - Issue #94
- Description: `loginHandler` in `services/shieldx-auth/auth-service/main.go` issues JWTs for any non-empty username/password and grants admin role to the `admin` username. With default deploy, anyone can mint valid bearer tokens.
- Location: `loginHandler` function inside the auth service main file.
- Recommendation: Integrate with a real credential store (hashed passwords/IdP), enforce rate limiting/MFA for privileged roles, and fail closed when Redis/session backends are unavailable. Add tests for valid/invalid credential flows.
- Done when: Incorrect credentials return 401, stored hashes or IdP APIs back authentication, and automated tests enforce the behaviour.

#### 55) Database restore command injection via `sh -c` - Issue #95
- Description: `shared/shieldx-common/pkg/database/backup.go` builds `gunzip -c <backupPath> | psql ...` using `fmt.Sprintf` and executes through `sh`. An attacker controlling the backup path or DB settings can inject shell payloads.
- Location: `BackupManager.Restore` function, plain SQL branch when `isCompressed` is true.
- Recommendation: Avoid `sh -c`; stream gzip contents via Go (`gzip.NewReader`) or call `pg_restore`/`psql` with `exec.CommandContext` argument slices. Validate and whitelist paths.
- Done when: Restore uses argument-safe invocations, injection strings are rejected, and tests cover malicious paths.

#### 56) Admin Marketplace unauthenticated mutation APIs - Issue #96
- Description: `services/shieldx-admin/marketplace/main.go` exposes `/packages/publish`, `/bounties/create`, `/bounties/submit`, etc., over plain HTTP without auth. Attackers can register rogue packages or consume resources.
- Location: Marketplace service handlers defined in the main file.
- Recommendation: Require JWT/mTLS for every mutating endpoint, enforce per-tenant RBAC, validate payloads (e.g. signed packages), and add rate limits/audit logging.
- Done when: Requests without valid credentials return 401/403, automated tests cover auth failures/successes, and rate/abuse controls are in place.

#### 57) Autoheal incident API unauthenticated - Issue #97
- Description: `services/shieldx-sandbox/autoheal/main.go` exposes `/autoheal/incident` without any authentication or rate limiting. An external actor can fabricate incidents that trigger `MeshController` to launch replacement VMs via `triggerRecovery`, leading to resource exhaustion or unauthorized infrastructure changes.
- Location: `services/shieldx-sandbox/autoheal/main.go` (handler registration) and `shared/shieldx-common/core/autoheal/mesh_controller.go` (`HandleIncident`/`triggerRecovery`).
- Recommendation: Require strong auth (JWT/mTLS + RBAC) for incident submission, validate node identifiers, enforce quotas, and ensure recovery workflows verify origin before provisioning resources.
- Done when: Unauthorized requests get 401/403, abuse tests cannot spawn incidents, and audit logs capture authenticated operators only.

#### 58) CDEFNET accepts any bearer token - Issue #98
- Description: `services/shieldx-sandbox/cdefnet/api.go` `authenticate()` only checks that the `Authorization` header has a `Bearer ` prefix and length >=32; it never verifies against a secret or signature. Attackers can craft arbitrary tokens and ingest/query threat intel under forged tenants.
- Location: `services/shieldx-sandbox/cdefnet/api.go` (functions `authenticate`, `submitIOCHandler`, `queryIOCHandler`).
- Recommendation: Integrate with real authentication (shared secret, JWT, or mTLS), bind tenants to validated identity, and reject unsigned tokens. Add unit tests exercising invalid/forged tokens.
- Done when: Forged tokens return 401, tenant identity maps to a trusted issuer, and tests enforce signature verification.

#### 59) Locator token issuance open to unauthenticated clients - Issue #99
- Description: The Locator service (`services/shieldx-gateway/locator/main.go`) exposes `/issue` without any authentication. Anyone can POST arbitrary `tenant/scope` data to mint valid locator tokens signed with the service key, bypassing downstream authorization.
- Location: `handleIssue` and `Run` in `services/shieldx-gateway/locator/main.go` (no auth middleware before handler).
- Recommendation: Gate `/issue` behind strong auth (mTLS + admission secret/JWT), enforce tenant allowlists, and log/alert on issuance. Consider separating public introspection from privileged issuance endpoints.
- Done when: Unauthenticated issuance attempts fail, token minting restricted to authorized callers, and coverage tests verify the control path.

#### 60) Continuous Auth scorer ships with hardcoded HMAC key - Issue #100
- Description: `NewHighPerformanceScorer(nil)` in `services/shieldx-auth/contauth/main.go` instantiates the scorer without providing a key, triggering the default `shieldx-production-hmac-key-2024` in `services/shieldx-auth/contauth/high_performance_scorer.go`. Anyone with repo access knows the HMAC secret protecting hashed biometrics, so they can recompute hashes, deanonymize telemetry, or craft cross-tenant correlation attacks.
- Location: `services/shieldx-auth/contauth/main.go` (call site) and `services/shieldx-auth/contauth/high_performance_scorer.go` (`NewHighPerformanceScorer`).
- Recommendation: Load a per-environment secret from KMS/secret manager, rotate regularly, and fail-fast if the key is missing. Add startup integration tests to ensure deployments never fall back to the built-in constant.
- Done when: ContAuth refuses to boot without a supplied key, the key loads from secure storage, and tests confirm the default is unreachable.

### N. Plugin / Sandbox / Runner Issues (new)

#### 61) Plugin Registry publish/run endpoints lack authentication - Issue #101
- Description: `shared/shieldx-sdk/plugin_registry/main.go` exposes `/plugins/publish`, `/plugins/run` and `/plugins/` without any authentication or RBAC. An attacker who can reach the service can upload arbitrary WASM artifacts or request plugin execution, potentially abusing internal artifact stores or triggering code paths.
- Location: `shared/shieldx-sdk/plugin_registry/main.go` (handler registration and handlers `publishPlugin`, `runPlugin`).
- Recommendation: Require JWT/mTLS + RBAC for publish and run operations. Enforce tenant/owner mapping: only owners or signed CI pipelines can publish. Add audit logging and rate limiting.
- Done when: Unauthorized calls return 401/403; owner binding is enforced and tests validate unauthorized attempts are rejected.

#### 62) WASM upload reads entire file into memory without explicit service-level caps - Issue #102
- Description: `publishPlugin` uses `r.ParseMultipartForm(32 << 20)` then `io.ReadAll(wasmFile)` which can still allocate the file content into memory and enable OOM for large uploads or multipart tricks. The code checks the WASM magic but doesn't apply a firm cap before reading.
- Location: `shared/shieldx-sdk/plugin_registry/main.go` (function `publishPlugin`).
- Recommendation: Enforce a strict maximum read (e.g., use `io.LimitedReader` or `r.Body = http.MaxBytesReader` before parsing), validate `Content-Length`, and reject oversized uploads with 413. Add CI fuzzer for multipart boundary abuse.
- Done when: Uploads > configured size return 413; memory usage is bounded in stress tests.

#### 63) Plugin-run executes a mock WASM payload instead of selected artifact - Issue #103
- Description: `runPlugin` constructs `mockWasmData := []byte("\x00asm\x01\x00\x00\x00")` and executes it rather than the actual plugin artifact fetched by `ArtifactID`. This both hides execution of real plugin artifacts (gap in functional path) and may allow attackers to trigger execution paths that are not exercised in production testing.
- Location: `shared/shieldx-sdk/plugin_registry/main.go` (`runPlugin` handler).
- Recommendation: Execute the real artifact retrieved from the artifact store after verifying permissions. If mock execution is required in dev/test, gate it behind a dev-only flag and never enable in prod.
- Done when: `runPlugin` executes actual artifact bytes after verification; dev-only mocks are disabled in prod builds and tested accordingly.

#### 64) Cosign CLI wrapper uses temp files and exec without input validation - Issue #104
- Description: `shared/shieldx-common/pkg/policy/cosigncli.go` writes digest/sig into temporary files and shells out to `cosign`. If the cosign binary path is attacker-controlled or PATH manipulated, this can lead to arbitrary command execution or use of a malicious cosign. Temp files are created in default temp dir and removed, but race conditions or broad perms may allow attackers to interfere.
- Location: `shared/shieldx-common/pkg/policy/cosigncli.go` (functions `Sign`, `Verify`).
- Recommendation: Validate `c.Path` (absolute path whitelist) or use a bundled cosign binary with fixed path. Use open file descriptors with secure permissions (0600) and avoid exposing predictable temp names. Where possible, replace exec-of-external-binary with in-process verification library (go-containerregistry/cosign). Add CI checks for PATH injection scenarios.
- Done when: Cosign invocation uses an audited path or in-process library; temp files are created with secure perms and path collision/test cases are added.

#### 65) Cosign keyless/default `KeyRef` empty allows uncontrolled verification - Issue #105
- Description: `CosignCLI` accepts empty `KeyRef` which defers to cosign keyless verification or environment identity. In environments without strict attestation, this can allow signature acceptance without a trusted key and weaken supply-chain guarantees.
- Location: `shared/shieldx-common/pkg/policy/cosigncli.go` (`CosignCLI.Verify` / `Sign`).
- Recommendation: Require a configured `KeyRef` or `COSIGN_KEY` in prod profiles and fail-fast if not present. For keyless workflows, enforce an allowlist of trusted OIDC issuers and enforce verification policies in CI. Document and gate keyless usage.
- Done when: Prod fails to start without `KeyRef` or an equivalent trusted verifier configured; CI enforces signature provenance rules.

#### 66) Sandbox dockerRunner executes host shell command with payload interpolation - Issue #106
- Description: `shared/shieldx-common/pkg/sandbox/sandbox.go` creates `cmd := []string{"/bin/sh", "-lc", fmt.Sprintf("printf %q | sha256sum", payload)}` and passes it to the container config, effectively interpolating payload into a shell command. If payload contains crafted sequences, and the container runtime or shell interprets them unexpectedly, it can lead to command injection or altered behavior. Additionally, using sh -lc inside container bypasses stricter execution models.
- Location: `shared/shieldx-common/pkg/sandbox/sandbox.go` (dockerRunner.Run).`
- Recommendation: Avoid `sh -lc` with formatted strings. Pass the payload as container args or via stdin/file, or use Go API to write a file into the container and compute hash in a controlled binary. Sanitize/whitelist inputs and reduce privileges (no shell access). Add unit tests for payloads containing shell-metacharacters.
- Done when: Containers are started with explicit argv (no `-c`), or payloads are provided via files/stdin; tests show metacharacter payloads do not change behavior.

#### 67) Temporary file pattern for WireGuard private key can leak to host - Issue #107
- Description: In `services/shieldx-gateway/ingress/main.go` the code runs a shell snippet that creates a temporary file (via `mktemp`) and writes a base64-decoded private key into it before passing to `wg set`. If an attacker can influence `privB64` or the temp dir, a private key could be written to host filesystem with insecure perms and survive beyond the command lifetime.
- Location: `services/shieldx-gateway/ingress/main.go` (mktemp/wg private-key shell snippet around line ~1469).
- Recommendation: Avoid writing private keys to host files. Use ephemeral file descriptors with restrictive perms (0600), write within a locked tmpfs, or use `wg` APIs that accept keys via stdin/fd if available. Ensure immediate secure deletion and avoid using shell pipelines where base64-decoded content lands in a world-readable temp file.
- Done when: Private keys are never materialized on shared host tmp directories; key material is handled through secure OS primitives and tests verify no leftover files.

#### 68) No authentication on Plugin Registry run path + policy enforcement gaps - Issue #108
- Description: `runPlugin` uses `plugin.SandboxPolicy` fetched from DB and passes it directly to `WasmRunner.ExecutePlugin`. If an attacker can publish or modify plugin metadata (see Issue #101), they could request a permissive sandbox policy (network/filesystem enabled) and then the runner will apply it.
- Location: `shared/shieldx-sdk/plugin_registry/main.go` (`runPlugin`) and `services/shieldx-sandbox/sandbox/runner/wasm_runner.go` (`ExecutePlugin` uses provided `SandboxPolicy`).
- Recommendation: Enforce server-side policy whitelists and never trust policy fields from untrusted DB inputs without RBAC. Validate and clamp sandbox policy server-side (e.g., force NetworkAccess=false unless owner has explicit privilege). Harden WasmRunner to apply least-privilege defaults and cap memory/CPU regardless of requested policy.
- Done when: Server enforces policy limits; attempts to request elevated policy are rejected with 403 and tests cover policy escalation attempts.

#### 69) No strong sandbox runtime policy enforcement — relying on exported test functions - Issue #109
- Description: `WasmRunner.TestPluginIsolation` checks for exported functions `test_network` and `test_filesystem` to detect isolation failures. This is a weak approach: malicious modules can omit these functions but still perform network/filesystem I/O via syscalls or WASI imports. Reliance on plugin-provided tests is insufficient for enforcement.
- Location: `services/shieldx-sandbox/sandbox/runner/wasm_runner.go` (`TestPluginIsolation` and `ExecutePlugin`).
- Recommendation: Enforce sandbox policies at runtime by disabling WASI host functions that provide network/filesystem access, using a restricted runtime configuration (policy-level restrictions), or running plugins inside hardened OS-level sandboxes (gVisor, Firecracker). Do not rely solely on plugin self-tests. Add runtime introspection and syscall/blocking observers.
- Done when: Runtime enforces no-network/no-filesystem regardless of plugin content; negative tests (attempt network call) fail deterministically.

#### 70) No explicit permission/owner validation when fetching plugin metadata - Issue #110
- Description: `runPlugin` fetches `plugin := pr.validator.GetPlugin(req.PluginID)` and uses `plugin.SandboxPolicy` and other metadata without verifying the requestor's authorization to run that plugin. This allows any caller to run artifacts belonging to other owners if endpoints are unauthenticated.
- Location: `shared/shieldx-sdk/plugin_registry/main.go` (`runPlugin` handler).
- Recommendation: Require caller authentication and enforce owner/tenant mapping before allowing execution. Add RBAC checks, owner-only run permission, and token-scoped claims to tie requests to owners. Add audit logs for plugin execution requests.
- Done when: Requests from non-owners receive 403 and automated tests confirm enforcement.

### O. File handling, bundle, and Firecracker issues (additional)

#### 71) Firecracker runner writes execution script to world-executable temp path - Issue #111
- Description: `shared/shieldx-common/pkg/sandbox/firecracker_runner.go` creates a script under `os.TempDir()` with a predictable name (`sandbox-<ts>.sh`) and writes it with mode `0500`. Predictable temp file names and use of host tmpdir can allow local users to race or inspect files; the script includes the raw payload content before VM isolation, exposing potential secrets and local escalation.
- Location: `shared/shieldx-common/pkg/sandbox/firecracker_runner.go` (`executeInVM`, scriptPath and os.WriteFile call).
- Recommendation: Use `os.CreateTemp` to get a safe, unique temp file, enforce file mode 0600, write under a locked tmpfs or chroot, and avoid embedding raw payloads on the host (stream into VM). Remove predictable naming and ensure immediate secure deletion; add tests for race conditions.
- Done when: Temp files use `os.CreateTemp`, secure perms set, and tests show no leftover file or predictable names.

#### 72) Firecracker `payloadPath` written with permissive perms (0644) - Issue #112
- Description: `shared/shieldx-common/pkg/sandbox/firecracker.go` / `firecracker_runner.go` writes payload files with `0644` permissions (world-readable), potentially exposing sensitive payloads to other local users/processes before or after execution.
- Location: `shared/shieldx-common/pkg/sandbox/firecracker.go` and `.../firecracker_runner.go` (writes to `payloadPath`/scriptPath with 0644/0500 previously observed).
- Recommendation: Use `0600` for payload files, write to secure tmp dirs accessible only to service UID, and purge files immediately. Consider in-memory streaming into the VM rather than materializing files on host.
- Done when: No file writes with non-private perms remain in sandbox code; test validates lack of readable files by other UIDs.

#### 73) Policy bundle WriteZip may include path traversal / zip-slip entries - Issue #113
- Description: `shared/shieldx-common/pkg/policy/bundle.go` writes files into the zip using the relative paths directly from `b.Files` without normalizing or validating for `..` or absolute paths. A crafted bundle could cause a consumer to extract files outside the intended directory (zip slip) if the bundle is unzipped elsewhere.
- Location: `shared/shieldx-common/pkg/policy/bundle.go` (`WriteZip` function, iterating `paths` and `zw.Create(p)`).
- Recommendation: Sanitize paths (reject entries containing `..` or absolute roots), normalize with `filepath.Clean` and ensure entries are contained under a single directory prefix. Add unit tests for zip-slip cases.
- Done when: `WriteZip`/unzip routines reject traversal entries and tests cover extraction boundary cases.

#### 74) WriteSignature writes sig files as world-readable (0644) - Issue #114
- Description: `WriteSignature` stores signature files with mode `0644`. Signatures and artifacts may be sensitive in some workflows and should not be world-writable/readable if in shared storage.
- Location: `shared/shieldx-common/pkg/policy/bundle.go` (`WriteSignature` function).
- Recommendation: Use `0600` and allow configuration for repository-managed storage; ensure signature dirs are access-controlled. Add CI lint to flag 0644 for signature files.
- Done when: Signature writes use `0600` by default and tests validate permissions.

#### 75) Presence of NoopSigner/NoopVerifier demo backdoors in policy pipeline - Issue #115
- Description: `NoopSigner` and `NoopVerifier` exist inside `shared/shieldx-common/pkg/policy/bundle.go` and accept unsigned/naive signatures. If accidentally wired into a prod path (misconfiguration), this completely breaks supply-chain signing guarantees.
- Location: `shared/shieldx-common/pkg/policy/bundle.go` (types `NoopSigner`, `NoopVerifier`).
- Recommendation: Remove Noop implementations from production builds or gate them behind build tags/dev flags. Fail-fast if the configured signer/verifier is a noop in prod profile. Document and block via CI detection.
- Done when: Noop implementations cannot be selected in prod and CI rejects bundles relying on them.

#### 76) LoadFromDir reads arbitrary files from disk referenced in manifest without path validation - Issue #116
- Description: `LoadFromDir` reads files from disk based on `mf.Policies` and uses `filepath.Join(dir, rel)` without canonicalization checks. A malicious manifest could reference files outside the intended dir (e.g., `../secrets`) and cause secret leakage when building bundles.
- Location: `shared/shieldx-common/pkg/policy/bundle.go` (`LoadFromDir`).
- Recommendation: Canonicalize and verify that `filepath.Join(dir, rel)` is within `dir` (use `filepath.Abs` + prefix check). Reject entries containing `..` or absolute paths. Add manifest validation tests.
- Done when: `LoadFromDir` rejects traversal entries and tests confirm secrets cannot be included by relative paths.

#### 77) Default embedded DB URLs with credentials in multiple services - Issue #117
- Description: Several services embed default `DATABASE_URL` values containing credentials like `postgres://user:pass@localhost/...` in code (for example in `shared/shieldx-sdk/plugin_registry` and `services/shieldx-sandbox/cdefnet/main.go`). If deployed with env misconfiguration, these defaults enable easy access.
- Location: multiple `main.go` files (examples: `shared/shieldx-sdk/plugin_registry/main.go`, `services/shieldx-sandbox/cdefnet/main.go`, `services/shieldx-deception/shadow/main.go`).
- Recommendation: Remove hardcoded DSNs; require `DATABASE_URL` to be explicitly set in prod profiles. Fail-fast on missing DB env. Document dev-only examples in test fixtures, not in runtime binaries.
- Done when: No default DSN with credentials remains in runtime main files; missing DB env triggers startup fatal in prod.

#### 78) Several os.WriteFile calls use permissive file modes for sensitive data - Issue #118
- Description: Multiple `os.WriteFile` calls persist data with 0644 (world-readable), including policy signatures, model metadata, and others. This may leak sensitive artifacts if files are stored on shared hosts or volumes.
- Location: `shared/shieldx-common/pkg/policy/bundle.go`, `shared/shieldx-common/pkg/ml/model_registry.go`, `shared/shieldx-common/pkg/policy/opa_route_test.go` (tests), and other locations.
- Recommendation: Use `0600` for sensitive files and evaluate where files get stored (avoid world-readable defaults). Add a code review checklist and CI lint rule to flag `os.WriteFile(..., 0o644)` usage for sensitive file names.
- Done when: Codebase uses secure perms for secrets/artifacts; lint rule passes.

#### 79) Locator uses os.CreateTemp in `persistRevocations` with default perms in data dir - Issue #119
- Description: `services/shieldx-gateway/locator/main.go` uses `os.CreateTemp("data", "revoke-*.tmp")` to write revocation snapshots. If the `data` directory is shared or has broad permissions, temporary files may be readable during the write window.
- Location: `services/shieldx-gateway/locator/main.go` (`persistRevocations`).
- Recommendation: Use `os.CreateTemp` with secure perms and then atomically rename; ensure `data` directory perms are restrictive (0700). Use `0600` for snapshot files. Add tests that verify file perms in CI environment.
- Done when: Revocation files are created with private perms and tests confirm no world-readable snapshots.

### P. System commands, backup, and validator issues (additional)

#### 80) WireGuard mesh config executes system commands using unsanitized inputs - Issue #120
- Description: `shared/shieldx-common/pkg/wgmesh/mesh.go` calls `exec.Command` for `ip` and `wg` operations using values that can come from `MeshConfig` (`InterfaceName`, `Address`, `Peers`, `PersistentKeepalive`) without strict validation. Malformed or attacker-controlled values could break command semantics or cause unexpected behavior on the host.
- Location: `shared/shieldx-common/pkg/wgmesh/mesh.go` (function `SetupMesh`, `TeardownMesh`, `GenerateKeyPair`).
- Recommendation: Strictly validate/whitelist interface names, CIDR addresses, port numbers, and peer fields before use. Prefer using programmatic APIs where available, avoid shelling out when possible, and run these operations in a least-privilege context. Add unit tests with malicious values.
- Done when: Inputs are validated/whitelisted and unit tests show rejected malformed values.

#### 81) PluginValidator does not write wasm/signature to temp files before calling cosign/trivy — verification ineffective - Issue #121
- Description: `shared/shieldx-common/core/fortress_bridge/plugin_validator.go` builds temp file paths for verification (`/tmp/plugin_<ts>.wasm`, `/tmp/plugin_<ts>.sig`) but never writes the provided `wasmData` or `signature` into those files before invoking `cosign`/`trivy`. This makes the verification path ineffective and may allow unverified artifacts to be treated as verified.
- Location: `shared/shieldx-common/core/fortress_bridge/plugin_validator.go` (`verifyCosignSignature`, `runTrivyScan`).
- Recommendation: Write the wasm bytes and signature into securely-created temp files (using `os.CreateTemp`), set restrictive perms, fsync, then invoke external tools against those files. Fail verification if writing fails. Add unit tests to ensure verification actually inspects provided bytes.
- Done when: Temp files are written and verified in tests; verification fails for invalid/missing sigs.

#### 82) PluginValidator invokes external binaries without controlling PATH or verifying binary identity - Issue #122
- Description: `plugin_validator` calls `cosign` and `trivy` via `exec.Command` relying on PATH resolution. If PATH is attacker-controlled or the host has malicious binaries, plugin validation and scanning could be subverted (PATH hijack). The code also logs tool output without sanitization.
- Location: `shared/shieldx-common/core/fortress_bridge/plugin_validator.go` (`verifyCosignSignature`, `runTrivyScan`).
- Recommendation: Use absolute, whitelisted binary paths or verify binary signatures. Prefer in-process libraries (cosign Go libs, Trivy libraries) to avoid shelling out. Validate tool outputs and scrub logs for secrets.
- Done when: Validator uses safe binary paths or in-process verification and CI tests simulate PATH hijack to ensure protection.

#### 83) Backup manager exports DB password via PGPASSWORD environment variable to child processes - Issue #123
- Description: `shared/shieldx-common/pkg/database/backup.go` sets `PGPASSWORD` in the child process environment to pass credentials to `pg_dump`/`psql`. Environment variables may be visible to local users or process inspection tools, and child process output may leak them into logs.
- Location: `shared/shieldx-common/pkg/database/backup.go` (`Backup`, `Restore`).
- Recommendation: Use `.pgpass` files with 0600 perms or use libpq connection strings with secure methods, avoid exporting passwords into the global environment, and minimize logging of child process output. Document secure backup/restore procedures and rotate creds after backup operations if possible.
- Done when: Backup/restore use secure credential mechanisms and tests confirm no password leakage via env or logs.

#### 84) Backup Restore builds shell command with gunzip piped into psql incorrectly (shell invocation risk) - Issue #124
- Description: In `Restore`, compressed SQL restore uses `args = append(args, "-c", fmt.Sprintf("gunzip -c %s | psql %s", backupPath, strings.Join(args, " ")))` and then calls `exec.CommandContext(ctx, "sh", args...)` which constructs a shell command with user-controlled `backupPath` and other args — a risk of shell injection or incorrect argument quoting (this is fragile and may already be exploited in other restore flows).
- Location: `shared/shieldx-common/pkg/database/backup.go` (`Restore`).
- Recommendation: Avoid shell pipelines. Decompress programmatically using gzip readers and stream into `psql` via stdin or use `pg_restore`/`psql` with args and explicit stdin redirection via `cmd.StdinPipe()` to avoid a shell. Sanitize `backupPath` inputs and add unit tests for malicious paths.
- Done when: No shell-based pipeline is used; compressed restores are implemented without `sh -c` and tests cover injection attempts.

#### 85) Child process output is returned/logged directly and may leak sensitive data - Issue #125
- Description: Several functions capture `CombinedOutput()` or `cmd.CombinedOutput()` and include the result in error messages or logs (e.g., `pg_dump` failures in `Backup`, `trivy`/`cosign` outputs in validator). These outputs can include credentials, file paths, or other sensitive data and may be stored in logs or returned to callers.
- Location: `shared/shieldx-common/pkg/database/backup.go`, `shared/shieldx-common/core/fortress_bridge/plugin_validator.go`.
- Recommendation: Scrub outputs before logging (redact passwords/paths), avoid including raw child output in errors sent to external callers, and store full outputs only in secure audit stores with restricted access.
- Done when: Logs do not contain sensitive output from child processes; CI checks detect accidental leaks.

#### 86) GenerateKeyPair uses external `wg genkey` and returns private key as plain string - Issue #126
- Description: `shared/shieldx-common/pkg/wgmesh/mesh.go` runs `wg genkey` and returns the private key as a Go string. If the caller logs or persists this string unsafely, it can leak private key material. Generating keys via external binaries also relies on PATH and external tool correctness.
- Location: `shared/shieldx-common/pkg/wgmesh/mesh.go` (`GenerateKeyPair`).
- Recommendation: Generate key material using in-process crypto libraries where possible, avoid returning secrets as plain strings (use secure buffers), and ensure callers persist keys to secure keystores with access controls. Validate and restrict where keys may be logged.
- Done when: Key generation uses secure primitives and key material is handled/stored via secret manager or secure file permissions.

#### 87) Predictable temp file names in validator/trivy path allow symlink/TOCTOU attacks - Issue #127
- Description: `plugin_validator` constructs predictable temp file names under `/tmp` using timestamps (e.g., `/tmp/plugin_<ts>.wasm`) for validation and scanning. Predictable temp names are vulnerable to symlink attacks and TOCTOU races in multi-user systems.
- Location: `shared/shieldx-common/core/fortress_bridge/plugin_validator.go` (`verifyCosignSignature`, `runTrivyScan`).
- Recommendation: Use `os.CreateTemp` to atomically create temp files with secure perms, avoid predictable names, and validate ownership/permissions before invoking external tools. Add tests simulating symlink attack scenarios.
- Done when: Temp files use `os.CreateTemp` and TOCTOU/symlink tests fail when attempted.

#### 88) Guardian service binds to 0.0.0.0 exposing internal execution API - Issue #128
- Description: `services/shieldx-forensics/guardian/main.go` now computes `addr := fmt.Sprintf(":%d", port)` and passes it to `http.ListenAndServe`, which listens on all interfaces. The code comment still claims the service is loopback-only behind ingress, but this change makes `/guardian/execute`, `/guardian/status`, and `/guardian/report` reachable anywhere the container port is published. Remote callers can submit arbitrary sandbox payloads, probe MASQUE/WCH relays, or brute-force the credit system without auth.
- Location: `services/shieldx-forensics/guardian/main.go` (binding logic near the bottom of `main()`).
- Recommendation: Restore loopback-only binding (`127.0.0.1`) or enforce strong authentication/TLS before exposing these endpoints. Combine with network policies/firewall rules to restrict access to ingress components, and add tests or lint rules to prevent accidental 0.0.0.0 bindings on privileged services.
- Done when: Guardian rejects connections from external hosts (connection refused or authenticated), and tests verify binding/auth constraints.

#### 89) ThreatGraph writer ships with hardcoded Neo4j admin credentials - Issue #129
- Description: `services/shieldx-forensics/threatgraph/writer.go` falls back to `NEO4J_USER=neo4j` and `NEO4J_PASSWORD=password` when environment variables are unset. Deployments that forget to override these defaults leave the Neo4j instance exposed with publicly known credentials, allowing attackers to dump or tamper with forensic graph data.
- Location: `services/shieldx-forensics/threatgraph/writer.go` (`getEnv("NEO4J_USER", "neo4j")` / `getEnv("NEO4J_PASSWORD", "password")` inside `main`).
- Recommendation: Remove production defaults for database credentials, fail fast if env vars are missing, and require secret provisioning through vaults. Provide sample creds only in development manifests/tests.
- Done when: Writer refuses to start without explicit credentials and integration tests confirm deployments cannot rely on weak defaults.

#### 90) Unauthenticated arbitrary Cypher execution via /graph/query - Issue #130
- Description: The ThreatGraph writer exposes `GET /graph/query?cypher=...` and passes the user-supplied string directly into `session.ExecuteRead` without validation or auth. Anyone reaching the service can run arbitrary Cypher against Neo4j, exfiltrate graph contents, or mutate data by embedding write clauses, bypassing ingest controls entirely.
- Location: `services/shieldx-forensics/threatgraph/writer.go` (`QueryGraph`, using `r.URL.Query().Get("cypher")`).
- Recommendation: Remove raw query endpoint or gate it behind authenticated, RBAC-protected tooling. If kept for admins, parse allow-listed parameters and reject queries containing write keywords. Add tests ensuring unauthorized callers receive 401/403.
- Done when: Production deployments do not expose free-form Cypher execution to unauthenticated clients.

#### 91) Relationship type injected unsafely into Cypher MERGE - Issue #131
- Description: `CreateEdge` builds the Cypher statement with `fmt.Sprintf("MERGE (from)-[r:%s]->(to)", rel.Type)`, allowing attackers to inject arbitrary Cypher when `rel.Type` contains backticks or clause delimiters. This enables arbitrary graph mutations or data exfiltration even if /graph/query were removed.
- Location: `services/shieldx-forensics/threatgraph/writer.go` (`CreateEdge`).
- Recommendation: Parameterize relationship type by validating against a whitelist (letters, digits, underscore) or use `fmt.Sprintf("`%s`", sanitizedType)`. Reject unexpected characters and cover with unit tests.
- Done when: Edge creation sanitizes relationship labels and tests show crafted inputs are rejected.

#### 92) MASQUE QUIC client disables TLS verification - Issue #132
- Description: Guardian’s MASQUE relay sets `tlsConf := &tls.Config{InsecureSkipVerify: true}` before dialing remote QUIC servers. An attacker positioned between Guardian and the MASQUE backend can impersonate the relay, intercept encrypted job traffic, or return crafted responses to abuse UDP relays.
- Location: `services/shieldx-forensics/guardian/main.go` (`masqueSingleExchange`).
- Recommendation: Enable proper TLS verification (set RootCAs/ServerName), pin certificates, or use mutual TLS. Add tests ensuring the client rejects self-signed/invalid certs.
- Done when: MASQUE connections fail against forged certificates and only trusted relays are accepted.

#### 93) Guardian UDP relay allows unauthenticated SSRF and reflection - Issue #133
- Description: `/wch/recv-udp` accepts attacker-controlled `udpReq.Target` and blindly relays packets via either MASQUE or direct UDP (`directUDPSingleExchange`). With the service bound to 0.0.0.0, any remote user can craft UDP traffic into the internal network (e.g., DNS, NTP) or abuse Guardian as an amplification proxy.
- Location: `services/shieldx-forensics/guardian/main.go` (`wch/recv-udp`, `directUDPSingleExchange`).
- Recommendation: Require authentication and enforce allow-lists for reachable UDP destinations. Consider disabling the feature unless explicitly needed and add unit tests for target validation.
- Done when: Requests targeting unapproved hosts/ports are rejected and relay endpoints demand auth.

#### 94) Rate limiter bypass via spoofed X-Forwarded-For - Issue #134
- Description: `makeRLLimiter` trusts the `X-Forwarded-For` header without verification. Attackers can supply a different header value per request, bypassing the per-IP limit and flooding Guardian’s sandbox queue.
- Location: `services/shieldx-forensics/guardian/main.go` (`makeRLLimiter`).
- Recommendation: Apply rate limiting on the actual remote address unless a trusted gateway sets the header, and enforce a allow-list of proxy IPs. Add tests showing spoofed headers no longer bypass limits.
- Done when: Requests with arbitrary `X-Forwarded-For` values are still throttled.

#### 95) Job map unbounded by default enabling remote DoS - Issue #135
- Description: Guardian tracks jobs in an in-memory map with `jobMax := getenvInt("GUARDIAN_JOB_MAX", 10000)` and a 600s TTL. With no auth, an attacker can submit thousands of jobs per minute, consuming memory and keeping the circuit breaker engaged, effectively denying service to legitimate analysts.
- Location: `services/shieldx-forensics/guardian/main.go` (job manager around `jobMax`, TTL cleanup loop, `/guardian/execute`).
- Recommendation: Enforce authentication, lower defaults, and reject requests when queue depth exceeds a safe threshold. Add alerting and tests covering over-capacity behavior.
- Done when: Excess jobs are rejected quickly and tests confirm the queue cannot exceed safe bounds unauthenticated.

#### 96) Credits consumption allows tenant hijacking - Issue #136
- Description: `/guardian/execute` lets callers supply arbitrary `tenant_id` values. `consumeCredits` forwards that ID to the Credits service without verifying caller identity, so attackers can drain or block other tenants by repeatedly submitting jobs with the victim’s ID.
- Location: `services/shieldx-forensics/guardian/main.go` (`consumeCredits` usage in `/guardian/execute`).
- Recommendation: Authenticate requests and bind tenant IDs to caller credentials (e.g., JWT claims). Reject mismatched IDs and add integration tests ensuring one tenant cannot charge another.
- Done when: Guardian enforces tenant identity binding and cross-tenant credit drains are impossible.

#### 97) Sequential job IDs leak full sandbox output to any caller - Issue #137
- Description: Guardian assigns predictable IDs (`j-1`, `j-2`, …) and `/guardian/status/{id}` serializes the entire `job` struct. Once a job completes, the response includes the full `Output` string, threat scores, errors, and backend details. Attackers can enumerate IDs and exfiltrate other tenants’ sandbox outputs, leaking investigation data and potential malware samples.
- Location: `services/shieldx-forensics/guardian/main.go` (`nextID`, `jobs` map, `/guardian/status/` handler).
- Recommendation: Require auth, randomize job IDs, and redact sensitive fields unless requested by the job owner. Add tests ensuring unauthorized enumeration attempts return 403 or redacted data.
- Done when: Job data is scoped to authenticated owners and enumeration no longer leaks outputs.

#### 98) Marketplace API lacks authentication allowing unauthorized package manipulation - Issue #138
- Description: All endpoints in `services/shieldx-admin/marketplace/main.go` (`/packages/publish`, `/packages/purchase`, `/bounties/*`, etc.) are exposed with `http.HandleFunc` and no authentication. Any network client can publish fake packages, drain bounty funds, or spoof purchases, undermining supply chain integrity and revenue accounting.
- Location: `services/shieldx-admin/marketplace/main.go` (HTTP handlers `handlePublish`, `handlePurchase`, `handleCreateBounty`, `handleSubmitSolution`, etc.).
- Recommendation: Require authenticated callers (JWT/API key) and enforce RBAC (only maintainers can publish, only verified tenants can purchase). Add tests to ensure unauthenticated requests receive 401/403.
- Done when: Marketplace endpoints reject unauthenticated callers and authorization checks prevent arbitrary package lifecycle mutations.

#### 99) Credits service disables auth by default exposing financial APIs - Issue #139
- Description: `withAuth` in `services/shieldx-credits/credits/main.go` returns `next` when `CREDITS_API_KEY` is unset. In default deployments the key is empty, so purchase/consume/reserve endpoints are unauthenticated, letting attackers mint and transfer credits across tenants.
- Location: `services/shieldx-credits/credits/main.go` (`withAuth`).
- Recommendation: Fail closed—require a non-empty API key or other auth (mTLS/JWT) before starting. Add startup checks and integration tests verifying unauthenticated calls are rejected.
- Done when: Service refuses to boot without configured credentials and unauthenticated requests receive 401.

#### 100) Rate limiter keyed by attacker-controlled header enables flood of credit operations - Issue #140
- Description: `rateLimitMiddleware` uses `key := r.URL.Path + ":" + tenant` where `tenant` comes from `X-Tenant-ID`. An attacker can bypass throttling by rotating this header each request, overwhelming the ledger without hitting the 20/min bucket.
- Location: `services/shieldx-credits/credits/main.go` (`rateLimitMiddleware`).
- Recommendation: Tie rate limiting to a trusted identity (e.g., authenticated tenant ID or remote IP behind trusted proxy) and ignore spoofed headers. Add tests showing crafted `X-Tenant-ID` values no longer bypass limits.
- Done when: Requests cannot evade throttling by manipulating headers and excessive bursts are rejected.

#### 101) Payment processing stub accepts any token allowing unlimited credit minting - Issue #141
- Description: `processPayment` in the credits service only checks that `PaymentMethod`/`PaymentToken` are non-empty before returning success with a fabricated reference. Combined with missing auth, attackers can POST arbitrary payloads to `/credits/purchase` and receive credits without real payment settlement.
- Location: `services/shieldx-credits/credits/ledger.go` (`processPayment`, used by `PurchaseCredits`).
- Recommendation: Integrate with a real PSP or at least validate signatures/webhooks; reject purchases until payment is verified. Add integration tests ensuring bogus tokens fail.
- Done when: Purchases require verifiable payment proof and fake tokens no longer increment balances.

#### 102) Auth service accepts any username/password granting tokens - Issue #142
- Description: `loginHandler` logs in any request with non-empty `username`/`password` (TODO left unimplemented). Attackers can mint access/refresh tokens for arbitrary identities and escalate by requesting the `admin` username.
- Location: `services/shieldx-auth/auth-service/main.go` (`loginHandler`).
- Recommendation: Implement real credential validation (hashed passwords, identity store) and lock out brute force. Add tests that bogus creds fail.
- Done when: Only legitimate accounts receive tokens and tests prove anonymous logins are rejected.

#### 103) Hard-coded OAuth client secrets in auth service encourage reuse in production - Issue #143
- Description: OAuth clients (`shieldx-web-app`, `shieldx-mobile-app`) embed `ClientSecret` strings in source. If ops reuse defaults, secrets are public, enabling token theft via client credential replay.
- Location: `services/shieldx-auth/auth-service/main.go` (OAuth2 provider initialization).
- Recommendation: Load client credentials from secure storage and block startup if defaults remain. Provide sample configs but ensure production builds fail on demo secrets.
- Done when: Secrets reside in vault/ENV only and automated checks catch default values.

#### 104) MASQUE QUIC server uses self-signed cert and no client auth - Issue #144
- Description: `services/shieldx-gateway/masque/main.go` generates a self-signed TLS cert on every boot and accepts any QUIC client. There is no authentication or trust anchor, making MITM trivial and letting attackers impersonate the relay for Guardian or other services.
- Location: `services/shieldx-gateway/masque/main.go` (`generateInsecureTLSConfig`, `quic.ListenAddr`).
- Recommendation: Use certificates signed by a trusted CA (or RA-TLS) and require client authentication/authorization before relaying traffic. Add tests ensuring unknown clients are rejected.
- Done when: MASQUE refuses unauthenticated clients and uses verifiable TLS identities.

#### 105) MASQUE relay allows arbitrary UDP targets creating open proxy - Issue #145
- Description: The MASQUE handler trusts the client-provided `udpTarget.Addr` and dials it via `net.DialUDP` without validation. Anyone reaching the QUIC port can tunnel UDP to internal services or participate in reflection attacks (e.g., DNS amplifiers).
- Location: `services/shieldx-gateway/masque/main.go` (`handleConn`).
- Recommendation: Enforce allow-listed destinations/ports, authenticate callers, or disable dynamic targets. Add negative tests covering disallowed addresses.
- Done when: Relay refuses unauthorized UDP targets and unauthenticated clients cannot proxy traffic.

#### 106) Shadow policy rollout trusts Noop verifier letting attackers push unsigned bundles - Issue #146
- Description: `/apply` accepts requests with a bare `digest`. When no URL is provided it calls `policy.VerifyDigest(policy.NoopVerifier{}, digest, []byte(digest))`, which always succeeds. Attackers can submit arbitrary digests to advance rollout without valid signatures.
- Location: `services/shieldx-policy/policy-rollout/main.go` (`/apply` handler, fallback digest path).
- Recommendation: Remove the Noop verifier in production. Require Cosign (or another signer) and reject digests lacking a verifiable signature. Add tests showing unsigned payloads fail.
- Done when: Policy rollout only accepts bundles with valid signatures and unsigned digests are rejected.

#### 107) Autoheal incident endpoint unauthenticated triggers spoofed recovery actions - Issue #147
- Description: `services/shieldx-sandbox/autoheal/main.go` exposes `/autoheal/incident` without auth. Posting arbitrary JSON kicks off recovery workflows (spawning replacement instances, marking incidents resolved), enabling attackers to disrupt telemetry, exhaust resources, or mask real outages.
- Location: `services/shieldx-sandbox/autoheal/main.go` (`http.HandleFunc("/autoheal/incident", ...)`).
- Recommendation: Require authenticated requests with signed incident payloads, add anti-automation guards, and validate incidents against trusted observability data.
- Done when: Only authorized control planes can submit incidents and spoofed calls are rejected.

 ====================================================================

#### 108) Decoy manager log path traversal allows arbitrary file write - Issue #148
- Description: `/analyze` accepts attacker-controlled `decoyId` and writes the analysis output to `filepath.Join("data", "sandbox", fmt.Sprintf("%s-<ts>.log", decoyId))` without sanitizing path segments. Crafted IDs containing `..` or absolute prefixes let callers escape `data/sandbox` and create or overwrite files elsewhere on the host.
- Location: `services/shieldx-deception/decoy-manager/main.go` (handler for `/analyze`, `path := filepath.Join("data", "sandbox", fmt.Sprintf("%s-%d.log", req.DecoyID, time.Now().Unix()))`).
- Recommendation: Normalize and validate IDs (e.g., allow `[a-zA-Z0-9_-]` and reject any path traversal tokens). Use `filepath.Clean`, enforce directory containment, or store artifacts under per-ID directories created with `os.MkdirTemp`.
- Done when: Attempts with `../` or absolute paths fail, files remain confined to `data/sandbox`, and tests cover traversal payloads.

#### 109) Decoy manager spawn/analyze endpoints unauthenticated enable container abuse - Issue #149
- Description: The decoy manager (`/spawn`, `/analyze`, `/list`) is exposed over HTTP without any auth. When `DECOY_DOCKER=1`, `/spawn` lets anyone on the network launch arbitrary Docker containers on the host. `/analyze` forwards attacker payloads to `sandbox.NewFromEnv()`—with sandbox builds (`sandbox_docker`/Firecracker) this executes code under the service account, compounding Issue #93. Even in noop mode, unbounded requests exhaust CPU/disk.
- Location: `services/shieldx-deception/decoy-manager/main.go` (handler setup in `main`, lack of auth guard around `/spawn`, `/analyze`, `/list`).
- Recommendation: Require strong authentication (Admission header, JWT, or mTLS), add per-tenant quotas/rate limits, and fail closed if secrets are missing. Consider gating Docker/sandbox execution behind feature flags restricted to trusted operators.
- Done when: Unauthenticated calls receive 401/403, container spawn requests are limited to authorized tenants, and tests confirm remote users cannot trigger sandbox execution or resource exhaustion.

#### 110) CDEFNET rate limiter uses unsynchronized map causing remote DoS - Issue #150
- Description: `APIServer.rateLimiter` is a plain `map[string]*RateLimiter` that gets mutated from every request in `rateLimit()` without any locking. Concurrent writes (which an attacker can trigger by sending parallel `/v1/submit-ioc` or `/v1/query-ioc` requests) lead to Go’s `fatal error: concurrent map writes`, crashing the process and taking the threat-intel feed offline.
- Location: `services/shieldx-sandbox/cdefnet/api.go` (struct `APIServer`, function `rateLimit`).
- Recommendation: Guard the map with a `sync.Mutex`/`sync.RWMutex`, or switch to `sync.Map`/token-bucket primitives that are concurrency-safe. Add fuzz or load tests that pound the endpoints in parallel to ensure the limiter no longer panics.
- Done when: Parallel requests no longer crash the service, rate-limit state updates run under synchronization, and stress tests confirm stability.

#### 111) Shapeshifter reflection map panics under concurrent requests - Issue #151
- Description: The shapeshifter `/reflect-exploit` handler calls `service.reflection.CreateMirror`, which writes to `ReflectionEngine.mirrors` (a plain map) with no synchronization. Parallel requests (easy to trigger remotely) perform concurrent map writes, causing Go to panic with `fatal error: concurrent map writes` and terminate the deception service.
- Location: `services/shieldx-deception/shapeshifter/main.go` (`CreateMirror`, `/reflect-exploit` handler).
- Recommendation: Protect the map with `sync.Mutex` or `sync.RWMutex`, or use a concurrency-safe structure. Add load tests that issue many concurrent `/reflect-exploit` requests to ensure the server remains stable.
- Done when: Concurrent requests no longer crash the service, reflection state updates are synchronized, and regression tests cover the parallel execution path.

#### 112) Shapeshifter reflection stores unbounded attacker payloads - Issue #152
- Description: Every `/reflect-exploit` call appends the attacker-supplied `payload` bytes into `AttackerMirror.Exploits` without any size cap or eviction. The JSON decoder will happily accept multi-megabyte base64 blobs, so a remote user can upload large payloads (or many unique `attacker_ip` keys) to exhaust heap memory and crash the service.
- Location: `services/shieldx-deception/shapeshifter/main.go` (`AttackerMirror.ReflectExploit`, `ReflectionEngine.CreateMirror`).
- Recommendation: Enforce strict request/body limits (`http.MaxBytesReader`), reject payloads above a few KB, and cap per-attacker history with bounded queues. Consider persisting only hashes/metadata instead of raw payload bytes.
- Done when: Oversized payloads are rejected, memory usage remains bounded under stress, and tests verify the limits.
