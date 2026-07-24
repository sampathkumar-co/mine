# Security policy

## Supported versions

Director OS 1.x receives security fixes. Pre-1.0 development revisions are unsupported after the v1.0 release.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, customer data, or private media. Use the repository's private GitHub security-advisory channel. Include:

- affected version or commit;
- reproduction steps;
- expected and observed impact;
- affected tenant or workspace boundary, when relevant;
- logs with secrets and personal data removed;
- suggested mitigation, when available.

Reports involving authentication bypass, cross-workspace access, signed-delivery forgery, payment or credit duplication, private-media disclosure, remote code execution, destructive deletion, or backup compromise should be treated as urgent.

## Operational expectations

Production operators must rotate all template secrets, require HTTPS and verified email, use Redis-backed fail-closed rate limiting, keep metrics protected, run Alembic migrations, store backups off-host, and complete `ops/go_live.sh` before public traffic is enabled.

Never commit production environment files, cloud credentials, Stripe keys, SMTP passwords, transcription-provider keys, customer footage, database dumps, or release-doctor reports containing operational details.
