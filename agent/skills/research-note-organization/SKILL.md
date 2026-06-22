---
name: research-note-organization
description: Structure evolving research ideas into maintainable, layered note systems that scale across brainstorming, literature review, and formulation phases. Use this skill whenever a researcher wants to organize growing notes, convert single documents into reference systems, reduce cognitive overhead from idea expansion, or establish a workflow that supports both exploration and clarity. Trigger on phrases like "how should I organize my notes?", "this document is getting unwieldy", "I'm accumulating too much research context", or when reviewing note-taking practices for active research projects.
compatibility: Logseq, Obsidian, GitHub + markdown, Notion (any system supporting hierarchical linking)
---

# Golden Rules for Research Note-Taking

## When to Use This Skill

You should use this skill when:
- Your research note has grown beyond a single comfortable document
- You're oscillating between high-level framing and implementation details without a clear structure
- You have multiple audiences (yourself, collaborators, eventual readers) for the same material
- You're discovering cross-cutting relationships between concepts but the linear format makes them hard to surface
- You want to reduce cognitive overhead while keeping the note "living" (actively updated)

---

## Core Principles

### Rule 1: Separate by Purpose, Not Chronology

**The Problem**: A single evolving document creates ambiguity about what's core vs. exploratory, what's settled vs. open.

**The Solution**: Organize into distinct layers:
- **Core** (the actual contribution): What problem are you solving? Why is it novel?
- **Mechanism** (how it works): Technical formulation, candidate approaches
- **Related Work** (existing solutions): Literature, competing ideas, inspirations
- **Open Questions** (unresolved gaps): Blocking issues, enrichment questions, deferred work
- **Implementation** (getting hands-on): How to test this, concrete next steps

Each layer is a separate document or folder. You reference *between* layers, not by copying.

---

### Rule 2: Use a Coupling Map as Your North Star

**The Problem**: As your idea expands, it's easy to lose sight of how concepts relate.

**The Solution**: Create a **single visual** (diagram, table, or graph) showing:
- **Concept nodes**: Key ideas in your research
- **Relationships**: How they connect ("X enables Y", "X contradicts Y", "X is a special case of Y")

This becomes the canonical reference. Every new note should connect to this map. When you discover a new relationship, you update the map first, then write the note.

**Format options:**
- Mermaid diagram (visual, easy to version control)
- Table with columns: `Concept | Definition | Connects to | Why`
- Graph in Obsidian/Logseq (native to these tools)

---

### Rule 3: Treat Open Questions as First-Class Objects

**The Problem**: Open questions scattered throughout your note create decision fatigue and make it hard to prioritize.

**The Solution**: Centralize them in a single document with metadata:
- **Blocking questions**: Must resolve before writing the paper
- **Enriching questions**: Nice-to-have, but not critical
- **Deferred questions**: Out of scope for this project

For each, include:
- The question itself
- Why it matters
- References to related notes
- Current status (open, has candidates, resolved)

This forces you to be explicit about what's stopping you from moving forward.

---

### Rule 4: Keep a Decision Log

**The Problem**: You make choices while exploring (e.g., "I'm focusing on domain randomization instead of parameter perturbation") but forget *why* later.

**The Solution**: Maintain a lightweight decision log:

```markdown
## Decision Log

**Date**: YYYY-MM-DD
**Question**: What modality of adversarial perturbation?
**Candidates**: [list]
**Decision**: [chosen approach]
**Reasoning**: [why this one wins]
**Status**: [Locked / Open / Revisiting]
```

This serves two purposes:
1. **For you**: Prevents re-litigating old decisions
2. **For readers**: Explains your design rationale

---

### Rule 5: Use Temporal Markers Selectively

**The Problem**: Timestamping every edit creates noise and false history.

**The Solution**: Only timestamp when:
- A **decision changed** (e.g., "Switched from approach A to approach B")
- A **major relationship surfaced** (e.g., "Realized PPO entropy collapse maps to loss landscape consolidation")
- A **blocking issue was resolved** (e.g., "Found SHAC-ASAM paper")

For routine updates (adding references, clarifying wording), skip the timestamp. This keeps the history signal-to-noise ratio high.

---

## The Folder Structure Template

```
Your Research Idea/
├── README.md (1 page max)
│   ├── One-liner pitch
│   ├── Why it matters
│   └── Links to key docs
│
├── Problem Formulation/
│   ├── gap-analysis.md (what's missing in existing work?)
│   ├── motivation.md (why should anyone care?)
│   └── framing.md (how are we framing the problem?)
│
├── Mechanism/
│   ├── core-insight.md (the "aha" moment)
│   ├── formulation.md (mathematical/technical statement)
│   ├── candidates.md (if multiple approaches exist)
│   └── analogies.md (mappings to other domains)
│
├── Related Work/
│   ├── [topic-1].md (e.g., implicit-curriculum.md)
│   ├── [topic-2].md
│   └── [topic-N].md
│   └── comparison-matrix.md (how your work sits relative to existing work)
│
├── Open Questions/
│   ├── PRIORITY.md (ranked by blocking status)
│   ├── [question-1].md
│   └── [question-2].md
│
├── Implementation/
│   ├── design-choices.md (parameter choices, hyperparameters, etc.)
│   ├── signals-and-thresholds.md (specific to RL/robotics work)
│   └── todo.md (concrete next steps)
│
├── Coupling Map/
│   ├── concepts-and-relationships.md (or .mermaid, or .svg)
│   └── updates.log (when you discover new relationships)
│
└── Decision Log/
    └── decisions.md (chronological, one entry per major choice)
```

---

## Workflow: Using This Structure

### Phase 1: Refactor (One-Time)
1. Read your existing note end-to-end
2. Extract **one core claim** (the contribution)
3. For each paragraph, ask: "Is this core? Background? Open question? Implementation detail?"
4. File each piece into the appropriate folder
5. Write a short coupling map connecting the pieces
6. Create the decision log by reviewing what choices you implicitly made

### Phase 2: Active Development
1. **Before adding something new**: Check the coupling map. Does this new concept belong? Does it connect?
2. **If exploring a new direction**: Add it to a candidate section or a new open question, *not* into core mechanism
3. **If you resolve something**: Move it from "open questions" to the relevant section, update the decision log
4. **Monthly**: Spend 15 min reviewing the coupling map. Are there new edges? Obsolete edges?

### Phase 3: Preparing to Write
1. **Outline the paper**: Use the folder structure as your outline (Problem → Mechanism → Related Work → Open Questions)
2. **Identify what's in vs. out**: Mark each section as "core" or "background" or "appendix"
3. **Write the paper**: Reference your structured notes section-by-section—no rework needed

---

## Quick Checklist

- [ ] **Separation by purpose**: Do your notes have distinct folders for Problem, Mechanism, Related Work, Open Questions?
- [ ] **Coupling map exists**: Is there a single document/diagram showing how concepts relate?
- [ ] **Open questions ranked**: Are blocking questions clearly distinguished from enriching ones?
- [ ] **Decision log**: Do you have at least one entry documenting a non-obvious choice?
- [ ] **No repetition**: If a concept appears in two documents, is one a reference to the other?
- [ ] **README**: Can you explain your research in 1 page? If not, your framing isn't sharp enough yet.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| **"Chronological sprawl"** — Adding new sections at the end every week | Creates archaeology—you have to dig through time to find a topic | Use folders instead. New related notes go in the same folder. |
| **"Repetition via copy-paste"** — Same concept explained in three places | When you refine your thinking, you have to update all three. One gets stale. | One authoritative source. Link everywhere else. |
| **"Everything is open questions"** — No distinction between "blocking" and "nice-to-have" | Decision paralysis. You don't know what to work on next. | Rank them. Lock down blocking decisions before writing. |
| **"Coupling map is internal"** — You have it in your head, not written down | New collaborators are lost. You forget relationships. | Make it a file. Update it when you discover new connections. |
| **"Timestamp every edit"** — Date stamps on every minor change | Signal drowns in noise. Hard to spot actual pivots. | Only timestamp decisions and "aha" moments. |

---

## Tips for Specific Domains

### For AI/ML Research
- **Mechanism**: Include candidate formulations (mathematical notation, algorithm pseudocode)
- **Related Work**: Separate "directly related" from "analogies from other fields"
- **Open Questions**: Distinguish "empirical unknowns" (need to run experiments) from "conceptual unknowns" (need theory)
- **Coupling Map**: Use nodes like "Loss Landscape", "Generalization", "Phase Transitions" and label edges with mechanism ("enabled by", "contradicted by")

### For Systems/Engineering Research
- **Mechanism**: Include design trade-offs (why did you choose X over Y?)
- **Implementation**: Link to actual code/config files, not just descriptions
- **Open Questions**: Separate "architectural" from "tuning"
- **Decision Log**: Especially important here—document why you chose each major component

### For Robotics
- **Mechanism**: Include diagrams of the control loop, formulation for contact dynamics
- **Related Work**: Track which papers use sim, which use real robots, which do transfer learning
- **Open Questions**: Highlight sim-to-real gaps explicitly
- **Coupling Map**: Nodes like "Simulation Fidelity", "Contact Modeling", "Domain Randomization"

---

## When to Refactor (Again)

Your structure worked, but now it's creaking. Signs to watch:

- You have more than **5 open questions**—some should move to a decision
- You have more than **1 README**—your core claim is splitting
- A single folder has **>5 files**—it needs sub-folders
- You're copying concepts between files—consolidate and link
- The coupling map has **>20 edges**—you're getting off track or it's time to write the paper

When you notice these, spend an hour refactoring. It's faster than wading through a mess later.

---

## References & Tools

**For organizing in markdown + links:**
- Obsidian (best for visual coupling maps)
- Logseq (best for hierarchical bullet-based notes)
- GitHub + markdown (best for version control and sharing)

**For coupling maps:**
- Mermaid (`graph LR` or `graph TD`)
- Graphviz
- Excalidraw (if you prefer hand-drawn style)

**For writing:**
- Use your folder structure as the paper outline
- Link from your draft back to the structured notes
- Notes are sources; paper is the synthesis

---

## Common Workflow Questions

**Q: When do I move something from "Open Questions" to "Mechanism"?**
A: When you have both (a) a candidate answer and (b) evidence that it works (even preliminary). Until then, it stays open.

**Q: Should every Related Work paper get its own file?**
A: No. Group by theme (e.g., `implicit-curriculum.md` covers 2-3 papers). A paper gets its own file only if it's so important it's referenced from multiple sections.

**Q: How detailed should the coupling map be?**
A: 5–15 nodes is ideal. If you have >20, either your research is too broad, or you're ready to write the paper. If you have <5, you might still be in pure brainstorming.

**Q: Can I share this structure with collaborators?**
A: Yes. Version control the entire folder structure. Collaborators can edit files in parallel, leave comments (via markdown or GitHub issues), and sync via git.

---

## Success Criteria

You'll know this is working when:

1. **Onboarding is fast**: A collaborator can read the README + coupling map and understand your research in 10 minutes
2. **Writing is easy**: Your paper outline matches your folder structure almost 1:1
3. **Decision-making is clear**: When you face a choice, you know exactly which open question it addresses
4. **Updates are localized**: When you refine something, you change one file, not three
5. **Growth feels manageable**: Adding new ideas doesn't feel chaotic; it just means a new file in the right folder

