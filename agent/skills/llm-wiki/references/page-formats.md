# Page Format Reference

Standard templates for wiki page types. These are starting points — adapt in SCHEMA.md for your domain.

---

## Source Summary Page

`wiki/sources/<slug>.md`

```markdown
# <Title>

**Type**: article | paper | book | transcript | note | other  
**Date**: YYYY-MM-DD  
**Author(s)**: ...  
**Source**: URL or citation  
**Ingested**: YYYY-MM-DD  

## Summary

2–4 sentence overview of the source and its main contribution.

## Key Claims

- Claim one.
- Claim two.
- ...

## Notable Details

Quotes, data points, examples worth preserving verbatim (attributed).

## Connections

- Supports: [[concepts/x]], [[entities/y]]
- Contradicts: [[sources/other-source]] on point Z
- Related: [[concepts/a]], [[sources/b]]
```

---

## Entity Page

`wiki/entities/<slug>.md`

```markdown
# <Name>

**Type**: person | organization | place | object | other  
**Also known as**: (aliases, if any)

## Overview

1–3 sentence description.

## Key Facts

- Fact one (source: [[sources/x]])
- Fact two (source: [[sources/y]])

## Role in This Wiki

Why this entity matters to the wiki's subject matter.

## Appearances

Sources that discuss this entity significantly:
- [[sources/a]] — context
- [[sources/b]] — context

## Connections

- Related entities: [[entities/x]], [[entities/y]]
- Related concepts: [[concepts/z]]
```

---

## Concept Page

`wiki/concepts/<slug>.md`

```markdown
# <Concept Name>

## Definition

Clear 1–3 sentence definition.

## Key Properties

- Property one
- Property two

## Evidence / Examples

What the wiki's sources say about this concept:
- [[sources/a]]: supports / describes / applies this concept by...
- [[sources/b]]: challenges this concept by arguing...

## Contradictions and Open Questions

Note unresolved tensions across sources.

## Connections

- Related concepts: [[concepts/x]], [[concepts/y]]
- Key entities: [[entities/z]]
```

---

## Query / Answer Page

`wiki/queries/<slug>.md`

```markdown
# <Question or Topic>

**Asked**: YYYY-MM-DD  
**Type**: comparison | analysis | synthesis | timeline | other

## Answer

The synthesized answer, in prose or structured form.

## Sources Consulted

- [[wiki/sources/a]]
- [[wiki/concepts/b]]

## Follow-up Questions

- ...
```

---

## Overview Page

`wiki/overview.md`

```markdown
# Overview

**Last updated**: YYYY-MM-DD  
**Sources ingested**: N  
**Wiki pages**: N

## Current Thesis / Big Picture

The evolving synthesis: what do all the sources say together?

## Major Themes

- Theme one: brief description, key pages [[concepts/x]]
- Theme two: ...

## Key Tensions

Unresolved contradictions or open questions across the wiki.

## Recent Updates

What changed in the last few ingests.
```
