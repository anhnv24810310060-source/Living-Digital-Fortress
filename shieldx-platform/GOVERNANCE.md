# Project Governance

## 1. Roles
| Role | Responsibilities | Requirements |
|------|------------------|--------------|
| Maintainer | Strategic direction, release approval, security response | Sustained high-quality contributions; consensus trust |
| Reviewer | Code review, triage issues/PRs | Active recent contributions |
| Contributor | Code / docs / tests | Signed DCO (if adopted) & follows guidelines |

## 2. Decision Making
* Default: Lazy consensus (silence >= 3 business days assumed agreement)
* Escalation: Simple majority of maintainers
* Security or embargoed issues: Private maintainer channel, need ≥2 maintainers to publish advisory

## 3. Adding / Removing Maintainers
* Nomination via PR updating `MAINTAINERS.md`
* Requires ≥2 existing maintainer approvals and no objections
* Inactivity (no meaningful contributions for 6 months) -> move to Emeritus after notice

## 4. Releases
* Semantic Versioning once API considered stable
* Release checklist includes: tests green, security scans pass, CHANGELOG updated, SBOM generated, signed artifacts

## 5. Security
* Coordinated disclosure per `SECURITY.md`
* CVE requests filed by a maintainer

## 6. Conflict Resolution
* Attempt direct discussion
* Escalate to neutral maintainer not involved
* Final vote among maintainers

## 7. Changes Requiring Explicit Approval
* Breaking API / wire format changes
* Licensing / legal notices
* Governance modifications
* Dependency additions with copyleft or uncommon licenses

## 8. Meetings (Optional / Future)
* Ad hoc; notes stored in `docs/meetings/` if established

## 9. Transparency
* All design proposals via public issues / PRs unless security-sensitive
