# The vault

A second brain for a solo founder who runs three small businesses. It is plain markdown, so it
opens in Obsidian or any editor, and it is written to be read by an agent in one hop.

**Northbeam, its businesses, clients and figures are fictional.** The method is the one I use
for my own businesses; the content is invented so it can be public.

## The rules

**1. One flat map.** [MAP.md](MAP.md) lists every note, grouped by business, with the question it
answers. A reader (human or agent) reads the map once and jumps straight to the right note. No
crawling folders, no chain of indexes.

**2. The vault answers, the source decides.** A figure in the vault is good enough to reason
with. Before publishing or charging it, check the source the note points to. The frontmatter
says how far to trust each note:

| `confidence` | Meaning |
|---|---|
| `verified` | checked against its source on the `updated` date |
| `snapshot` | copied from a source on the `updated` date; check before publishing |
| `unverified` | stated by the founder, no written source |

**3. Seven frontmatter keys, no more.**

```yaml
updated: 2026-10-20     # always
status: current         # current | superseded
type: offer             # index | decision | operations | offer | icp | acquisition | clients
                        # | platforms | communications | legal | archive
business: tidepool      # the top folder; shared for what crosses businesses
confidence: verified    # verified | snapshot | unverified
date: 2026-09-14        # decisions only: when it was decided
sources:                # where the figures come from
  - Stripe Billing catalogue
```

**4. Decisions are append-only.** One note per decision in `<business>/decisions/YYYY-MM-DD-slug.md`.
A new decision never deletes the old one: the old note becomes `status: superseded` and gains a
**Superseded by** link, so the history of why something changed is never lost.

**5. Relative markdown links, never wikilinks.** An agent reading the raw file can resolve
`[x](../tidepool/support.md)` without knowing Obsidian's index.

**6. The map of keys, never a key.** The vault says which tool, what for, with which account and
where the credential lives. The credential itself lives in the password manager.

**7. Closed clients only.** Leads and the pipeline live in the CRM; the vault keeps the clients
that signed and what was delivered.

`sba lint` checks every rule that a machine can check, including that no note contains anything
that looks like a credential.
