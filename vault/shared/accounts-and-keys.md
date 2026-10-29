---
updated: 2026-10-12
status: current
type: platforms
business: shared
confidence: verified
sources:
  - 1Password vault list and the rotation log
---
# Accounts and keys

The vault never stores a credential. It says where each one lives; the credential itself lives
in 1Password.

## Password manager

1Password, owned by the founder, with four vaults:

| Vault | What is in it |
|---|---|
| Northbeam · Shared | Google Workspace admin, Invoice Ninja, the bank, the domain registrar |
| Studio | n8n admin, the Hetzner console, the Attio API token |
| Tidepool | Supabase, Vercel, Stripe live, Resend, Sentry |
| Field Notes | Buttondown, Plausible |

The freelance designer has access to nothing but a shared Figma team; no vault is shared with
anyone.

## Rules

- Every admin account has two-factor authentication with the founder's hardware key. The
  recovery codes live in the 1Password item of each service.
- Production keys are rotated every six months. Last rotation: 2026-09-30.
- A key is never pasted into a chat, a ticket, a commit or a note. If one leaks, rotate it first
  and investigate after.
- Client credentials belong to the client: see [studio platforms](../studio/platforms.md).
