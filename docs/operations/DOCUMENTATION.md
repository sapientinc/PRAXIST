# Praxist Data-Collection Module: Technical Documentation

**Last updated:** 27 August 2026

**Purpose:** As required by Section 1.4 (Data Collection) of the Fair Source License Agreement (Version 1.0), this document describes the source-code structure of the data-collection module and discloses the data-receiving endpoint URL. It is written for users, enterprise IT, and security auditors.

**Related documents:** [Privacy Notice](../legal/PRIVACY.md); [Praxist User Data Collection Notice](../legal/product-usage-data-notice.md) (in-product notice text, Notice version 3)

> This document describes the current implementation. Keep it and the Privacy Notice aligned with changes to the product-usage contract.

---

## 1. Module Map

| Part | Path | Notes |
| --- | --- | --- |
| Client SDK | `praxist/product_usage/` | Consent management, environment identity, event generation, local outbound queue, upload |
| Client integration | `praxist/infrastructure/product_usage.py` | Observer that projects Research Run lifecycle into telemetry events; failures are isolated from the Research Run |
| CLI entry point | `praxist/cli/product_usage.py` | The `praxist product-usage` command family (notice / consent / status / withdraw) |
| Server Collector | `praxist/product_usage/app.py`, `collector.py`, `postgres.py`, `retention.py` | HTTP ingestion, schema validation, idempotent persistence, retention deletion |
| Server deployment | `services/product_usage/` | Dockerfile, Nginx configuration, Compose, deployment scripts |
| Protocol & schema | `praxist/product_usage/protocol.py`, `schemas/v2/*.json` | Closed Schema V2 (shared by client and server) |
| Legal text | `docs/legal/product-usage-data-notice.md` | The notice text shown in-product (Notice version 3) |

## 2. Client File Map

| File | Responsibility |
| --- | --- |
| `consent.py` | Consent-state storage: an `unset` / `granted` / `denied` state machine that fails closed; atomic writes (0600 permissions); consent records are bound to the Notice version — a version mismatch counts as no consent; Agent-assisted replies recognize only `Yes` / `Agree` / `No` / `Disagree` |
| `identity.py` | Environment identity: generated at random via UUIDv4 and persisted locally in `environment.json`; never derived from any personal, device, or task information |
| `paths.py` | Fixed per-OS local file paths (Section 5); no environment-variable or project-level overrides |
| `lifecycle.py` | Generation of run-level event IDs, telemetry run IDs, event sequence numbers, and the four lifecycle events |
| `outbox.py` | Bounded local SQLite outbound queue (offline buffering, later delivery, cleared on withdrawal) |
| `batching.py` | Bounded JSON batch encoding (parsed identically on the server) |
| `transport.py` | Endpoint selection and HTTP sending (Section 3) |
| `client.py` | The `UsageSdk` facade: every collection failure is isolated and never reaches the Research Run |
| `protocol.py` | Closed Schema V2 models and boundary constants (Section 4) |
| `notice.py` | Loads the in-product notice text |
| `app.py` / `collector.py` / `postgres.py` / `retention.py` | **Server side**: HTTP entry, validation and idempotency core, PostgreSQL persistence, retention-deletion job |

## 3. Data-Receiving Endpoint URLs

| Environment | Endpoint | Notes |
| --- | --- | --- |
| **Production (release builds)** | `https://telemetry.theaiscientist.com/v1/events` | HTTPS encryption, server certificate verification, no redirect following |
| **Development (internal `.dev` builds only)** | Internal development collector (address not published) | Plain HTTP; handles only internal development/test data, never user data; not shipped with release builds |

Endpoint selection: `default_batch_sender()` in `transport.py` chooses the sender based on whether the Praxist version string contains `.dev`. The production endpoint must be a valid HTTPS URL or construction is refused outright.

Request contract:

- `POST` with `Content-Type: application/json`; request body capped at 32 KB; at most 50 events per batch;
- carries only the fixed, protocol-level User-Agent `Praxist-Product-Usage/2` — no cookies and no additional request headers;
- network timeout is 2 seconds; a success response is `202` with body `{"accepted": n, "duplicates": n}`;
- error responses: `400` (malformed request / unsupported schema version), `413` (too large), `415` (non-JSON), `503` (ingestion paused or temporarily unavailable);
- transmission failures never block or affect the Research Run; undelivered events stay in the local outbound queue and are sent automatically once connectivity returns.

## 4. Closed Schema V2 and Boundary Constants

The schema is a closed model (`extra="forbid"`): both client and server reject any out-of-schema field, and schema extensions require an explicit change to `protocol.py` plus a version bump.

| Constant | Value | Meaning |
| --- | --- | --- |
| `SCHEMA_VERSION` | 2 | Event-structure version |
| `CONSENT_NOTICE_VERSION` | 3 | Current Notice version (consent records are bound to it) |
| `MAX_BATCH_EVENTS` | 50 | Maximum events per batch |
| `MAX_REQUEST_BYTES` | 32 KB | Maximum request-body size |
| `MAX_ERROR_SUMMARIES` | 16 | Maximum error-summary groups per event |
| `MAX_ERROR_COUNT` | 65,535 | Maximum count per error group |
| `MAX_DURATION_MINUTES` | 43,200 (30 days) | Maximum recorded active-run duration |

The closed value lists for event fields and error categories are in Section 3 of the [Privacy Notice](../legal/PRIVACY.md) and in `schemas/v2/usage-event.schema.json` and `schemas/v2/usage-batch.schema.json`.

## 5. Consent State and Local File Paths

All paths below are fixed, current-user-only locations with 0600/0700 permissions:

| Platform | Consent record | Outbound queue / environment identity |
| --- | --- | --- |
| Linux | `~/.config/praxist/product-usage/consent.json` | `~/.local/share/praxist/product-usage/outbox.sqlite3`, `environment.json` |
| macOS | `~/Library/Application Support/Praxist/product-usage/consent.json` | `outbox.sqlite3`, `environment.json` in the same directory |
| Windows | `%LOCALAPPDATA%\Praxist\product-usage\consent.json` | `outbox.sqlite3`, `environment.json` in the same directory |

- **Run-state files:** `runs/<hash>.json` alongside `environment.json`; stored locally only and never uploaded. **How the hash is computed** (see `run_state_path()` in `praxist/product_usage/paths.py`): the run-directory path is first normalized with `expanduser` and `resolve` (expanding the user directory and resolving it to a canonical absolute path), then UTF-8 encoded and hashed with SHA-256; the 64-character lowercase hexadecimal digest becomes the filename. SHA-256 is one-way, so the original path cannot be recovered from the filename, and since the file never leaves the machine, no path information is exposed;
- **CLI:** `praxist product-usage notice | consent | status --json | withdraw`;
- **First use:** the notice is shown and an explicit choice is awaited only in an interactive terminal while the state is `unset`; non-interactive environments remain `unset` (i.e., nothing is collected).

## 6. Offline and Failure Behavior

- **Offline or upload failure:** events are buffered in the local outbound queue and delivered when connectivity returns — **research functionality is entirely unaffected**;
- **Withdrawal (`withdraw`):** immediately stops all future capture and deletes every local unsent event;
- All client-side collection/upload exceptions are isolated by the fail-closed facade in `client.py` and never interrupt a Research Run.

## 7. Server-Side Privacy Measures

- Nginx reverse proxy: `access_log off`; strips the `X-Forwarded-For`, `X-Real-IP`, `Cookie`, and `User-Agent` headers before requests reach the application;
- Two-level rate limiting (per client and global); storage capacity ceiling `COLLECTOR_MAX_TABLE_BYTES` (default 2 GB) — new events are refused beyond it;
- Master ingestion switch `COLLECTOR_INGESTION_ENABLED`, which can pause ingestion entirely (returns 503);
- The Collector container binds to `127.0.0.1` only and reaches the managed PostgreSQL over a private endpoint; the database is never exposed to the public internet;
- `received_at` is generated by the server after validation and cannot be supplied or altered by the client;
- Retention job: runs once at service start and at least every 24 hours thereafter; events enter the deletion window on day 179 (one day of scheduling slack against the stated 180-day ceiling); if the retention job fails, it exits and the container restarts to retry;
- The server has **no** interface for deleting already-delivered events by environment identifier (a design constraint disclosed in Section 8 of the [Privacy Notice](../legal/PRIVACY.md)).

## 8. How to Audit It Yourself

1. See the exact fields that would be reported: `praxist/product_usage/schemas/v2/*.json` and `protocol.py`;
2. Check current consent status: `praxist product-usage status --json`;
3. Inspect local files: the paths listed above (all readable and writable only by the current user);
4. Packet-capture verification: the endpoint, the fixed User-Agent, and the request-body content can all be verified with standard network capture tooling;
5. The complete notice text: `praxist product-usage notice`, or `docs/legal/product-usage-data-notice.md` in the repository.

## 9. Versioning and Change Management

- The schema version, Notice version, and all boundary constants are defined centrally in `praxist/product_usage/protocol.py`;
- When the notice content changes, `CONSENT_NOTICE_VERSION` increases; previously recorded consent does not carry over to the new version and is requested again at first use;
- If the implementation described in this document changes, this document, the [Privacy Notice](../legal/PRIVACY.md), and the in-product notice must be updated together.

---

**Contact:** praxist@sapient.inc
