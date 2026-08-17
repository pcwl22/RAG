# Changelog

This project follows Keep a Changelog conventions. Release tags should use semantic versioning.

## Unreleased

### Added

- OIDC and API-key authentication with tenant-scoped RBAC.
- PostgreSQL row-level security and tenant-composite document keys.
- Release evaluation datasets, RAGAS quality gates, backup/restore tooling, SBOM, and image scans.

### Changed

- Production rate limiting fails closed when Redis is unavailable.
- RAG release baselines are bound to an implementation fingerprint.
- Database schema contracts use content-hashed SQL migrations.

### Security

- Upload validation, metrics authentication, least-privilege database roles, and default-deny write
  authorization are enabled for production profiles.
