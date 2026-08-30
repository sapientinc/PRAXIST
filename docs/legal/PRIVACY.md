# Praxist Privacy Notice (Product Usage Data)

**Last updated:** 27 August 2026

**Related documents:** [Fair Source License Agreement (Version 1.0)](https://github.com/sapientinc/praxist/blob/main/LICENSE.md);
[Praxist User Data Collection Notice](product-usage-data-notice.md) (Notice
version 3); [Product Usage Technical Documentation](../operations/DOCUMENTATION.md)

---

## 1. Overview

This Notice applies only to Praxist's **optional product-usage data collection**. The data controller is Sapient Intelligence Pte Ltd ("we", "us").

Core principles:

- **Voluntary.** Collection is predicated on your explicit consent and is never mandatory.
- **Revocable.** You may withdraw consent at any time; collection stops immediately upon withdrawal.
- **No impact on use.** Declining or withdrawing consent does not affect the installation of Praxist or any research functionality.
- **Pseudonymized and minimized.** We collect only pseudonymized lifecycle data within a closed schema — never any task content, input data, or research results.

## 2. Preconditions for Collection (Voluntary Consent Mechanism)

Praxist collects product-usage data only when **all three** of the following conditions are met:

1. the installed build contains an approved collection transport;
2. product-usage collection is enabled for that build; and
3. you separately and explicitly select **"Share product usage"** after this Notice is made available for review, or reply `Yes` / `Agree` to a consent prompt shown during installation.

Please note:

- Accepting the Praxist User Agreement or the Fair Source License Agreement does **not** constitute consent to data collection; the two are independent;
- when no choice has been made (status "unset"), **nothing is collected or uploaded**;
- in non-interactive environments (no terminal input), no consent prompt is shown and collection remains off;
- the same rule applies to Agent-assisted installation: only `Yes` / `Agree` (consent) and `No` / `Disagree` (refusal) are recognized; any other wording is treated as no consent given;
- this Notice is versioned (currently Notice version 3). When the Notice changes, its version number increases; consent you previously recorded does not carry over to a new version and will be requested again.

## 3. What We May Collect (If You Consent)

The product-usage protocol uses a **closed schema** (Schema V2): only the fields below are permitted, and any additional field is rejected by both the client and the server.

### 3.1 Common lifecycle fields

| Field | Description |
| --- | --- |
| `schema_version` | Version of the product-usage event structure (currently 2) |
| `praxist_version` | Public Praxist version |
| `consent_notice_version` | The Notice version you consented to |
| `environment_id` | Environment identifier: a random UUID generated locally, stable across Research Runs within one Praxist environment (see Section 5) |
| `telemetry_run_id` | Independent random identifier for a single Research Run |
| `event_id` | Random identifier for a single event, used for correlation and deduplication |
| `event_sequence` | Sequence number within the same Research Run |
| `event_type` | Lifecycle event type (see Section 3.2) |
| `occurred_at` | Client-side event time (UTC, to the second) |
| `error_summaries` / `error_summaries_truncated` | Bounded structured error-category counts, and whether the bounded list was truncated (see Section 3.3) |

### 3.2 The four lifecycle events

| Event | What it records |
| --- | --- |
| `run_started` | The generation ordinal, the planned Peer count, and aggregate counts of Peers in the planning, running, completed, cancelled, failed, and unknown states at the run-start boundary |
| `generation_finished` | The same fields recorded at a durable generation boundary (state counts sum to the planned Peer count; "completed" means only that a Peer lifecycle returned normally — it does not assert that any scientific result is valid) |
| `run_finished` | Active run duration in complete minutes (capped at 43,200 minutes, i.e. 30 days) and whether the cap was reached |
| `run_reconciled` | The same duration fields, recorded when a previously unfinished run is resumed and trustworthy terminal processing is later completed |

### 3.3 Structured error summaries

Each lifecycle event may carry at most 16 grouped error summaries. Each group may contain only the following closed values:

- `scope`: run / generation / peer
- `stage`: setup / launch / execution / finalization / reconciliation
- `error_type`: configuration / resource / orchestration / runtime / external_dependency / storage / unknown
- `error_code`: PRX-CAPACITY / PRX-PEER-LAUNCH / PRX-PEER-RUNTIME / PRX-RUNTIME / PRX-RUN-FAILED / PRX-UNKNOWN
- `reason_code`: auth_error / quota_exhausted / rate_limited / timeout / provider_unavailable / runtime_error / tool_unavailable / invalid_request / budget_denied / budget_expired / capacity_unavailable / process_start_failed / state_unreadable / unexpected_termination / unknown
- `count`: number of matching errors (capped at 65,535), plus whether the cap was reached

This structure is **technically incapable** of containing raw error messages, logs, stack traces, provider responses, or arbitrary text.

### 3.4 Time fields

- `occurred_at` is generated by the local Praxist client at a lifecycle milestone, converted to UTC using the local system clock (and may therefore reflect clock inaccuracy);
- `received_at` is added by the server-side Collector after validation and cannot be supplied or altered by the client. It represents arrival time and is used only for storage management and retention calculation.

## 4. What We Do Not Collect

Product-usage events do **not** include:

- research task content, prompts, research results, files, filenames, project paths, or commands;
- environment variables, API keys, saved login credentials, logs, stack traces, raw error messages, or arbitrary text;
- model names, service-provider names, provider responses, or account information;
- names, email addresses, operating-system details, hardware information, Python version, client time zone, cookies, or arbitrary request headers;
- individual Peer identities or individual Peer outputs;
- IP addresses (event bodies never contain them; the temporary handling of network connection information is described in Section 6).

## 5. Pseudonymization Methodology

Praxist uses **pseudonymization**, not full anonymization:

- **Generation.** `environment_id`, `telemetry_run_id`, and `event_id` are all randomly generated via standard UUIDv4 (`uuid4()`);
- **No derivation.** These identifiers are not derived from usernames, accounts, device serial numbers, MAC addresses, IP addresses, hostnames, project paths, or task content, and they are not simple incrementing sequences;
- **Persistence.** `environment_id` is generated the first time an environment needs one and stored locally in `environment.json` (readable and writable only by the current user, permission 0600); it remains stable across runs within that environment. `telemetry_run_id` and `event_id` are generated per run and per event;
- **Local path hashing.** Local run-state files are named with the SHA-256 hash of the run path; these files **stay on your machine and are never uploaded**;
- **Honest characterization.** Because `environment_id` is stable across runs, events from the same environment could in theory be linked to one another — this is pseudonymized data, not fully anonymous data. However, it cannot directly identify you or your device, and we commit to **never using it in any way** to identify an individual (consistent with our commitment in Section 7).

## 6. Network and Transmission

- **Production endpoint:** `https://telemetry.theaiscientist.com/v1/events` (HTTPS encryption, server certificate verification, no redirect following);
- Requests carry only a fixed, protocol-level User-Agent (`Praxist-Product-Usage/2`); each request is capped at 32 KB and at most 50 events per batch; network timeout is 2 seconds;
- Transmission failures never block or affect Research Runs; events that fail to send are stored locally and delivered later when the network is available (see Section 8);
- **Server-side handling of connection information.** Product-usage event bodies contain no IP addresses, cookies, or arbitrary request headers. Network services necessarily process connection information transiently to deliver requests, protect the service, and apply rate limits. The Collector deployment disables access logging and strips forwarded IP, cookie, and client User-Agent headers before application processing; none of them are persisted as product-usage event data.

## 7. Purposes of Use

Collected product-usage data is used **only** for:

1. analyzing product reliability;
2. improving Praxist's features, performance, and user experience; and
3. anonymous, aggregate-level statistics that are never presented in a way that could identify a particular user.

We commit that we will **not**:

- sell product-usage data to any third party;
- use it for advertising or marketing; or
- link it with any other dataset about you or your users in order to identify an individual.

## 8. Storage, Retention, and Deletion

- **Server-side storage.** Managed PostgreSQL, accessed over a private endpoint; no public internet database exposure;
- **Retention.** Delivered raw events are retained for **at most 180 days** from server receipt and are then deleted by a scheduled retention process (the job runs at least once daily; by design, events enter the deletion window on day 179, leaving one day of scheduling slack so the stated 180-day ceiling is never exceeded);
- **Local unsent events.** Stored in the local SQLite outbound queue (`outbox.sqlite3`, readable and writable only by the current user), cleared after successful delivery, and deleted immediately upon withdrawal;
- **Honest note.** The server does not offer an interface to delete **already-delivered** events by environment identifier. Withdrawal stops future collection and deletes local unsent events; delivered events are deleted automatically when the retention period expires.

## 9. Your Rights and How to Exercise Them

| Action | Command |
| --- | --- |
| View the complete Notice text | `praxist product-usage notice` |
| Check current consent status | `praxist product-usage status --json` |
| Record consent | `praxist product-usage consent` (or select "Share product usage" during first use) |
| **Withdraw consent at any time** | `praxist product-usage withdraw` |

- Withdrawal immediately stops all future capture and deletes all local unsent events;
- Declining or withdrawing **does not affect** the installation, operation, or any research functionality of Praxist;
- You may also exercise your rights of access, rectification, deletion, or complaint through the contact point in Section 13.

## 10. Data Recipient and Cross-Border Arrangements

- **Data recipient:** Sapient Intelligence Pte Ltd (the same legal entity as the Licensor under the license agreement)
- **Server location:** Johor, Malaysia
- **Cross-border transfer:** If you are located in mainland China and choose to opt in, your consent constitutes authorization for the transfer of the above data — which contains no names, contact details, accounts, IP addresses, or content data (see the closed lists in Sections 3 and 4) — to the Collector in Malaysia, limited to the fields enumerated in the Section 3 closed schema. Users in the EU/EEA are covered by Section 10.1.

### 10.1 Supplementary Notice for Users in the European Union and EEA (GDPR)

Praxist is distributed globally, and users in the EU/EEA may likewise choose to opt in to product-usage collection. Under the GDPR, pseudonymized data remains personal data, and we apply the following rules to such data:

- **Legal basis.** Your explicit consent only (GDPR Art. 6(1)(a)) — data collection is off by default and is enabled only when you actively select "Share product usage";
- **Withdrawal.** You may withdraw consent at any time via `praxist product-usage withdraw`; withdrawal does not affect the lawfulness of processing carried out on the basis of consent before its withdrawal (Art. 9(3));
- **Cross-border transfer.** Data is transferred to and processed by the Collector in Johor, Malaysia. This transfer relies on the explicit consent you give at opt-in (the derogation in GDPR Art. 49(1)(a)) and is limited to the fields enumerated in the Section 3 closed schema of this Notice;
- **Your rights.** Access, rectification, restriction of processing, data portability, objection, and erasure. Send access, rectification, or deletion requests to praxist@sapient.inc;
- **Honest note on erasure.** As described in Section 8, the server currently is not able to search or delete **already-delivered** events by environment identifier; delivered events are deleted automatically at most 180 days after receipt. You may include the `environment_id` from your local `environment.json` in a deletion request to verify ownership of the environment, and we will manually process your request after verification;
- **We do not:** sell personal data, use it for advertising or marketing, or carry out automated decision-making with legal or similarly significant effects;
- **Supervisory authority.** You have the right to lodge a complaint with the data protection supervisory authority of your member state.

## 11. Data Security

- Production traffic is HTTPS-encrypted end to end with certificate verification, and redirects are refused;
- Closed schema: both the client and the server reject any field outside the schema;
- The server enforces global and per-client rate limits and a storage capacity ceiling; a master ingestion switch can shut down the collection entry entirely in an emergency;
- Local consent records, the environment identifier, and the outbound queue are readable and writable only by the current user (0600/0700 permissions);
- Events are deleted automatically when the retention period expires — no action required from you;
- Security contact: praxist@sapient.inc

## 12. Changes to This Notice

We may revise this Notice and the in-product notice as the product evolves. When the notice content changes, the Notice version number increases; consent you recorded is valid only for the version it was given against, and renewed consent will be requested at first use after an upgrade. If this Notice and the in-product notice diverge, the version presented to you at the time of consent prevails.

## 13. Contact Us

- Privacy contact: praxist@sapient.inc
- Company: Sapient Intelligence Pte Ltd (incorporated in Singapore)
- Commercial licensing and license-agreement matters: see the contact details at the end of the Fair Source License Agreement (Version 1.0) (praxist@sapient.inc)
