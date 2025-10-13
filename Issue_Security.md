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
 
- Tài liệu/SOP:
  - Quy trình rotate secrets (JWT/Admission/Neo4j); chuẩn hóa CORS/headers; network policy cho /metrics và health.
