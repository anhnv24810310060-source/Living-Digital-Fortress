# Supply Chain: SBOM, Image signing, Reproducible builds

This repo includes minimal tooling to generate SBOMs, sign container images using Cosign (keyless via OIDC), and build reproducible binaries via GoReleaser snapshots.

## Local usage

- Generate SBOMs (CycloneDX JSON):
  - `make sbom-all` → outputs under `dist/sbom/`

- Sign an image (requires `cosign`):
  - Keyless (OIDC in your shell): `make image-sign IMAGE=ghcr.io/<org>/<repo>:<tag>`
  - With key ref: `make image-sign IMAGE=... KEY_REF=cosign.key`

- Reproducible snapshot build (requires `goreleaser`):
  - `make release-snapshot` → artifacts under `dist/` with trimpath and empty buildid.

## CI workflow

- `.github/workflows/supply-chain.yml` runs on push to `main` and manually via dispatch.
- Steps:
  - Generate SBOM for repo root and `ml-service/` using Syft.
  - Build snapshot binaries via GoReleaser.
  - Build, push, generate SBOM for all container images in `docker/` via a matrix; cosign-sign each image (keyless OIDC) after push.
  - Optionally cosign-sign an arbitrary image if `image` input is provided.
  - Upload SBOMs and snapshot artifacts.

### Permissions & Secrets

- Uses `id-token: write` to enable Cosign keyless in GitHub Actions.
- No extra secrets are needed for keyless. For registry logins, configure `docker/login-action` before signing if the image is private.

## Notes

- SBOM format is CycloneDX JSON to ease ingestion by common tools.
- GoReleaser config is minimal and currently builds `cmd/policyctl` only. Extend `builds:` to add more binaries as needed.
- For image provenance/SLSA, integrate `slsa-framework/slsa-github-generator` in a future iteration.

# Supply Chain Hardening (December)

This doc outlines SBOM generation, container image signing, and reproducible builds.

## SBOMs

- Tooling: Syft (CycloneDX JSON output)
- Make targets:
  - `make sbom-go` — scan Go workspace into `dist/sbom/sbom-go.json`
  - `make sbom-python` — scan `ml-service/` into `dist/sbom/sbom-python.json`
  - `make sbom-all` — both of the above
- CI: `.github/workflows/supply-chain.yml` uploads `dist/sbom/**` as artifacts.

## Image signing

- Tooling: Sigstore Cosign (keyless via GitHub OIDC by default).
- Usage:
  - `make image-sign IMAGE=ghcr.io/<org>/<app>:<tag>`
  - Optional: `KEY_REF=cosign.key` to sign with a key instead of keyless.
- CI: Matrix job builds/pushes images from `docker/` and cosign-signs them keylessly. You can also manually provide `image` input to sign any image.

## Reproducible builds

- Tooling: Goreleaser snapshot mode (no publishing), with `--clean` for deterministic artifacts.
- Usage: `make release-snapshot`
- CI: runs in `supply-chain.yml` on push to main.

## Prereqs

- Local: install `syft`, `cosign`, and `goreleaser`.
- CI: actions install these tools per job.

## Notes

- Keep images distroless and pinned. Prefer keyless signatures in CI.
- SBOMs use CycloneDX JSON for compatibility with most scanners.

## RA-TLS quick note (for January scope)

- Use `pkg/ratls` AutoIssuer in services for short-lived certs with SPIFFE SANs.
- Server: `issuer.ServerTLSConfig(requireClientCert=true, expectedTrustDomain=<your-domain>)`
- Client: `issuer.ClientTLSConfig()`
- Rotate every ~45m with 60m validity to meet "≤1h" lifetime target.

## Enforce verify in the cluster

- Policy controller manifest: `pilot/hardening/image-signing.yml`.
- Apply to your cluster (example):

```bash
kubectl create namespace shieldx-system
kubectl apply -f pilot/hardening/image-signing.yml
```

- Adjust identities in the policy to match your signing source:
  - issuer: GitHub OIDC issuer (e.g., https://token.actions.githubusercontent.com)
  - subject: repository pattern (e.g., your org/repo)

- After applied, any image matching the policy scope must present a valid cosign signature from the allowed identities.
