# ShieldX Policy Bundle Spec (v0 draft)

Goal: Provide a signed, reproducible, and verifiable package of OPA/Rego policies for rollout and drift detection.

## Manifest

- name: string
- version: string (SemVer)
- created_at: RFC3339
- opa_version: optional
- policies: string[] (relative paths within bundle)
- annotations: map<string,string>

## Canonicalization & Hashing

- Normalize manifest via JSON marshal (encoding/json) with stable field order.
- Concatenate manifest bytes with each file in sorted path order using the framing:
  - "\n--FILE--\n" + path + "\n" + content
- Hash with SHA-256. Digest (hex) is the bundle identity.

## Packaging

- Zip archive with:
  - manifest.json at root
  - policy files at their relative paths

## Signing

- Primary: Sigstore cosign (sign-blob / verify-blob)
  - Keyed mode: `--key <keyRef>` supports file/KMS.
  - Keyless mode: OIDC identity (CI) with transparency log.
- Signature artefact: raw signature bytes stored alongside bundle digest in registry.

## Verification

- Given manifest+files (or digest), verify signature via cosign verify-blob.
- Record result in metrics (verify_success_total/verify_failure_total) and logs.

## Rollout

- Canary 10% workloads. Rollback automatically if SLO thresholds breached (error rate/latency).
- Promote when healthy for a defined bake time.

## Drift Detection

- Periodically compute running hash and compare with registry.
- Emit Prometheus metrics + event logs when mismatch.

## Future

- DSSE envelope; attestation subject = bundle digest.
- SBOM for policies; provenance SLSA.
