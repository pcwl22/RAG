# Security Policy

## Supported versions

Security fixes are applied to the latest commit on `main`. Production deployments should use an
immutable image digest built from a reviewed release commit rather than a floating branch or tag.

## Reporting a vulnerability

Do not open a public issue for suspected vulnerabilities or exposed credentials. Use the
repository's GitHub **Security → Report a vulnerability** private advisory workflow. Include the
affected commit, deployment mode, reproduction steps, impact, and any suggested mitigation.

The maintainers should acknowledge a report within three business days, provide an initial
severity assessment within seven business days, and coordinate disclosure after a fix or accepted
mitigation is available. Never include production documents, access tokens, tenant identifiers, or
personal data in a report.

## Operational response

For a confirmed incident, revoke affected credentials, isolate the impacted tenant and workload,
preserve audit evidence, rotate IdP and service secrets, and rebuild from a reviewed commit. Follow
the backup and restore controls in `industrial-rag/deploy/production-checklist.md`.
