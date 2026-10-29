---
updated: 2026-10-12
status: current
type: platforms
business: studio
confidence: verified
sources:
  - 1Password vault "Studio"
---
# Studio platforms

| Tool | What for | Account | Where the credential lives |
|---|---|---|---|
| n8n, self-hosted on a Hetzner VPS in Falkenstein | runs every client automation | ops@northbeam.example | 1Password "Studio" → "n8n admin" |
| Hetzner | the VPS | ops@northbeam.example | 1Password "Studio" → "Hetzner console"; SSH only with the founder's hardware key |
| Attio | CRM and pipeline | founder@northbeam.example | Google sign-in; API token in 1Password "Studio" → "Attio API" |
| Google Workspace | mail, calendar, drive | founder@northbeam.example | 1Password "Northbeam · Shared" → "Google admin" |
| Cal.com | diagnosis bookings | founder@northbeam.example | Google sign-in |
| Invoice Ninja, self-hosted | invoices | billing@northbeam.example | 1Password "Northbeam · Shared" → "Invoice Ninja" |

## Client credentials

Each client's API keys live in the client's own accounts and are stored, encrypted, in that
client's n8n credentials. They are never copied into 1Password, a document or this vault.
