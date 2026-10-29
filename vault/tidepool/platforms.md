---
updated: 2026-10-12
status: current
type: platforms
business: tidepool
confidence: verified
sources:
  - 1Password vault "Tidepool"
  - Vercel project settings
---
# Tidepool platforms

| Service | What for | Account | Where the credential lives |
|---|---|---|---|
| Supabase, EU region (Frankfurt) | database, authentication, file storage | ops@tidepool.example | 1Password "Tidepool" → "Supabase"; the service-role key exists only there and in Vercel's environment variables |
| Vercel | hosting for the app and the public site | founder@northbeam.example | 1Password "Tidepool" → "Vercel" |
| Stripe | subscriptions, and studio payments through Stripe Connect | billing@tidepool.example | live secret key in Vercel's environment variable `STRIPE_SECRET_KEY` and in 1Password "Tidepool" → "Stripe live"; never in the repository |
| Resend | transactional email | ops@tidepool.example | 1Password "Tidepool" → "Resend" |
| Sentry | error tracking | ops@tidepool.example | 1Password "Tidepool" → "Sentry" |
| GitHub | the private repository `tidepool-app` | the founder's account | hardware key; no deploy keys |
