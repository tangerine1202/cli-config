---
name: llm-wiki
description: >
  Set up and operate a personal LLM-maintained wiki — a persistent, compounding knowledge base made of interlinked markdown files. Use this skill whenever the user wants to build a personal knowledge base, research wiki, reading companion, or any accumulated knowledge system where an LLM writes and maintains structured notes from sources. Trigger on phrases like "build a wiki", "knowledge base", "ingest this article/paper/source", "maintain notes with LLM", "personal wiki", "research notes", "reading notes", "summarize and file", or any request to organize accumulated knowledge into a persistent structure. Prefer this skill over ad-hoc summarization when the user's intent is to build something that compounds over time.
---

# LLM Wiki

A skill for setting up and operating a personal knowledge base where the LLM writes and maintains all content. You source; the LLM files, cross-references, and keeps everything current.

---

## Core Concept

Unlike RAG (retrieve-and-answer from raw files), this pattern has the LLM **compile knowledge once** into a structured wiki and keep it current as new sources arrive. The wiki is the persistent artifact — cross-references are already built, contradictions already flagged, synthesis already written. Every new source and every good answer makes it richer.

**Three layers:**
- `raw/` — immutable source documents (articles, papers, notes). Never modified.
- `wiki/` — LLM-generated markdown pages. You read; the LLM writes.
- `wiki/SCHEMA.md` — conventions and workflows for this wiki instance. Co-evolve with the user over time.

---

## First-Time Setup

When the user wants to start a new wiki, help them initialize the structure and write a SCHEMA.md tailored to their domain.

### Directory layout

```
<wiki-root>/
├── raw/                    # Source documents (immutable)
├── wiki/
│   ├── SCHEMA.md           # Conventions and workflows
│   ├── index.md            # Catalog of all wiki pages
│   ├── log.md              # Append-only operation log
│   ├── overview.md         # High-level synthesis (optional, grows over time)
│   ├── sources/            # One summary page per ingested source
│   ├── entities/           # Pages for people, places, organizations, objects
│   ├── concepts/           # Pages for ideas, themes, methods, frameworks
│   └── queries/            # Valuable answers filed back as pages (optional)
```

Adjust categories to the domain. A fiction reading wiki might use `characters/`, `locations/`, `themes/`. A research wiki might use `papers/`, `methods/`, `findings/`. Ask the user what categories fit.

### SCHEMA.md

Write a SCHEMA.md at `wiki/SCHEMA.md` that documents:
- **Purpose**: what this wiki is for
- **Category definitions**: what goes in each subdirectory
- **Page format conventions**: what sections each page type should have (see reference below)
- **Ingest workflow**: steps to follow when processing a new source
- **Query workflow**: how to answer questions and when to file answers back
- **Lint checklist**: what to check during periodic health passes

SCHEMA.md is the LLM's operating manual for this wiki. It should be read at the start of each session. Update it whenever conventions change.

### index.md and log.md

**index.md** — catalog of all wiki pages:
```markdown
# Index

## Sources
- [[sources/article-title]] — One-line summary. (2026-04-01)

## Entities
- [[entities/name]] — One-line description.

## Concepts
- [[concepts/topic]] — One-line description.
```

Update index.md on every ingest and whenever pages are added or renamed.

**log.md** — append-only chronological record:
```markdown
## [2026-04-01] ingest | Article Title
Processed. Updated 8 pages: sources/article-title, entities/x, concepts/y ...

## [2026-04-01] query | "What does X say about Y?"
Filed answer to queries/x-and-y.md
```

Each entry starts with `## [YYYY-MM-DD] <operation> | <label>` for easy grepping.

---

## Operations

### Ingest

When the user provides a source to process:

1. **Read** the source document fully.
2. **Discuss** (optional but recommended): surface key takeaways with the user before writing. Ask what to emphasize.
3. **Write summary page** at `wiki/sources/<slug>.md`. Include: title, date, source type, key claims, notable quotes (attributed), and links to related wiki pages.
4. **Update entity pages**: for each person, organization, place, or object mentioned significantly — create or update their page. Add what this source says about them.
5. **Update concept pages**: for each idea, method, or theme — create or update. Note if this source supports, contradicts, or nuances existing content.
6. **Update overview.md** if the new source meaningfully shifts the big picture.
7. **Update index.md**: add the new source page and any new entity/concept pages.
8. **Append to log.md**: record what was processed and which pages were touched.

A single source typically touches 5–15 wiki pages. This is expected and correct.

**Contradiction handling**: when new content contradicts an existing claim, note it explicitly on both pages:
```markdown
> ⚠️ Contradicts [[sources/earlier-source]]: that source claimed X; this source claims Y.
```

### Query

When the user asks a question against the wiki:

1. Read `wiki/index.md` to find relevant pages.
2. Read those pages in full.
3. Synthesize an answer with citations to wiki pages (e.g., `[[sources/title]]`).
4. **Decide whether to file**: if the answer involved non-trivial synthesis, comparison, or discovery — offer to save it as `wiki/queries/<slug>.md`. Good answers shouldn't disappear into chat history.
5. If filed, update index.md and log.md.

Output format is flexible: prose, comparison table, timeline, bullet list — whatever fits the question.

### Lint

Periodically (or when the user asks), health-check the wiki:

- **Contradictions**: flag claims on different pages that conflict.
- **Stale content**: flag pages that may have been superseded by newer sources.
- **Orphan pages**: pages with no inbound links from other wiki pages.
- **Missing pages**: important entities or concepts mentioned in multiple places but lacking their own page.
- **Missing cross-references**: pages that should link to each other but don't.
- **Data gaps**: topics the wiki covers shallowly that a web search or targeted source could fill.

Produce a lint report and suggest follow-up sources or questions to address gaps.

---

## Page Format Reference

See `references/page-formats.md` for standard page templates. Read it when writing pages for the first time in a session, or when the user asks about page structure.

---

## Session Start

At the start of each working session, read:
1. `wiki/SCHEMA.md` — conventions for this wiki
2. `wiki/index.md` — current state of the wiki
3. Last 5–10 entries in `wiki/log.md` — recent activity

This restores context without needing to re-read every page.

---

## Tips

- **Slugs**: use lowercase-hyphenated filenames. E.g. `entities/john-von-neumann.md`, `concepts/transformer-architecture.md`.
- **Links**: use `[[wiki-relative-path]]` or `[display name](relative-path.md)` syntax. The former renders in Obsidian; the latter is portable markdown.
- **The wiki is a git repo**: version history, branching, and collaboration come free.
- **Scale**: the index file is sufficient navigation up to ~200–300 pages. Beyond that, consider adding a simple search tool.
- **Don't over-engineer early**: start with a minimal schema and evolve it. The SCHEMA.md documents what actually works, not what was planned.
