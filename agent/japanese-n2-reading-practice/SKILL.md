---
name: japanese-n2-reading-practice
description: |-
  Turns a Japanese podcast or YouTube transcript into a full-length reading-immersion article written in JLPT N2 reading-section register (formal 説明文・論説文・随筆 style, no furigana, no vocab list), followed by a short bonus set of real N2-style 4-choice comprehension questions with a collapsible answer key. The article — long enough for genuine reading practice — is the main deliverable; the questions are a side benefit at the end.
  Trigger whenever the user gives a Japanese transcript and wants N2 reading practice, reading-immersion material, an "N2 reading article" for JLPT prep, or exam-format comprehension questions — even phrased as "make this into reading practice," "turn this transcript into an N2 article," or "quiz me on this in Japanese." Prefer this over the softer furigana learner-article workflow when the user trains for the N2/N1 reading section or wants exam-register text without reading aids. Also trigger for a full 5-genre N2 reading mock test from a transcript.
---

# Japanese N2 Reading Practice Generator

Turn a Japanese podcast/YouTube transcript into a **full-length article for reading immersion**,
written the way N2 reading-section passages are written — then add a small set of exam-style
comprehension questions at the end as a bonus.

The article is the point. The user reads it to build reading stamina and to get used to the
formal written register (説明文・論説文・随筆) that the N2 reading section is drawn from. The
questions are a light check at the end, not the main event — so never sacrifice article length
or richness to get to the questions faster.

This is the exam-register complement to the softer furigana learner-article workflow: it
deliberately uses no furigana and no vocabulary list, because reading unassisted N2-register
kanji text *is* the skill being trained. If the user wants furigana or a vocab table, that is the
other skill's job — point them there rather than bolting it on here.

## Why the transcript must be rewritten, not summarized or trimmed

A spoken transcript — especially a raw auto-generated one — is full of things that never appear
in written Japanese: greetings to the audience, self-promotion, fillers (えー、あの、なんか),
false starts, self-corrections, repeated points, and tangents. It is also often mis-transcribed
(wrong kanji, dropped particles). Your job is to **re-express the substance as a clean, coherent
written article**, not to lightly edit the speech. Think of it as an editor turning an interview
recording into a magazine essay: same ideas and voice, completely rebuilt prose.

Two failure modes to avoid:
- **Under-rewriting:** the article still sounds like a transcription — 皆さん, 〜んですよね,
  rhetorical asides to listeners, filler rhythm. If it does, it isn't done.
- **Over-compressing:** shrinking the whole episode to a 200-character exam snippet. That throws
  away the reading practice, which is the main value. Keep the substance and the length.

## Process

1. **Read the whole transcript** and pull out the substantive content: the actual argument,
   observations, facts, anecdotes, and any interesting specifics (numbers, examples, comparisons).
   Discard fillers, audience address, self-promotion, and tangents that don't serve the topic.
   Silently fix obvious ASR errors (wrong homophone kanji, dropped particles).

2. **Write the full-length article.** This is the main deliverable.
   - Rewrite into clean, formal **written** Japanese. Default to a polished 丁寧語 (です・ます)
     essay for reflective/personal topics, or だ・である for expository/argumentative ones —
     whichever matches the source's intent. Read `references/n2-genres.md` to match the register
     and rhetorical shape of real N2 passage types.
   - Give it real structure: a short lead, 3–5 developed body sections with headings, and a
     closing reflection. Preserve the depth of the source — most of the substantive ideas should
     survive, reorganized and cleaned, so the reader gets a satisfying, sustained read.
   - Length scales with the source, but aim for a genuine long read — roughly **900–1,800
     characters** for a typical 10–20 minute episode. Longer is fine when the source is rich; the
     goal is reading volume, not brevity.
   - Bold a handful of key words or set phrases inline where it aids the reader, but do **not**
     add furigana and do **not** append a vocabulary list.

3. **Add the bonus questions.** After the article, add a short `## 理解度チェック（N2形式）`
   section: 3–5 genuine N2-style questions, each with exactly 4 choices, mixing a couple of
   detail/meaning questions with at least one gist/inference question (筆者が最も言いたいこと,
   下線部の意味, 〜の理由). Distractors should be plausible misreadings, not absurd. See the real
   question stems per genre in `references/n2-genres.md`.

4. **Answer key as a collapsible block** so the reader can attempt the questions first. Use the
   Obsidian callout `> [!note]- 解答・解説` (the trailing `-` collapses it by default) and briefly
   explain why the right answer is right and why the closest distractor is wrong.

See `references/example-output.md` for the full target structure and a worked example.

## Output structure

```
# [Article title in Japanese] — [short English gloss]

> **出典 Source:** [channel / episode, from the transcript]
> **ジャンル Genre:** [説明文 / 論説文 / 随筆 …] （N2読解レベル）

---

[Full-length article: lead paragraph, then ## sections with formal written prose, then a
closing reflection. No furigana. No vocab list.]

---

## 理解度チェック（N2形式）

問1 … （4 choices）
問2 … （4 choices）
…

> [!note]- 解答・解説
> 問1 正解：X — …
> …
```

## Optional: full 5-genre mock test

If the user explicitly asks for a "mock test," "模擬試験," or "all the passage types," switch
modes: instead of one long article, produce all five real N2 reading genres (短文・中文・
統合理解・主張理解・情報検索) as separate exam-accurate passages at their real (shorter) lengths,
each with its own questions and answer key. `references/n2-genres.md` has the length, register,
and question conventions for each. This mode is exam-drilling, not reading immersion — only use it
when the user asks for it.

## Multiple transcripts

Produce one article per transcript by default. If two transcripts genuinely argue opposing sides
of the same topic and the user wants a comparison, they can be combined into a single 統合理解
mock-test item instead.
