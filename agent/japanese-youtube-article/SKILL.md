---
name: japanese-youtube-article
description: >
  Converts a Japanese YouTube (or podcast) transcript into a polished, in-depth
  Japanese-learner article: a clean written-Japanese rewrite that keeps EVERY point,
  statistic, and anecdote, expanded to give lots of exposure by deliberately recycling key
  vocabulary and grammar, with notable grammar patterns triple-bolded inline, plus a closing
  文法 section listing those patterns with a Traditional-Chinese (Taiwan / 繁體中文) nuance
  explanation and a correct-vs-incorrect usage example for each.

  Trigger whenever the user uploads or pastes a Japanese YouTube/video/podcast transcript and
  wants it turned into a readable article or study material — even if they don't say "skill"
  or "article." Key phrases: "make an article from this transcript," "clean up this
  transcript," "create learning material," "add grammar notes," "文法 list in 繁體中文," or
  simply attaching a Japanese transcript and asking to process it. It differs from a plain
  summary: the goal is a faithful, complete, expanded retelling, not a condensed summary.
---

# Japanese YouTube Transcript → Learner Article

Turn a raw Japanese YouTube/podcast transcript into a clean, well-organized article that a
Japanese learner (中級, N3–N2) can read comfortably, followed by a vocabulary-and-grammar
study section with Traditional-Chinese translations.

The transcript is **spoken, casual, and messy** (filler words, restarts, 話し言葉 like
「〜だよね」「むちゃむちゃ」「やばくない?」). Your job is to rewrite it into **polished
written Japanese** (丁寧語 / 敬体) while **preserving all of the content**. This is the
single most important thing about this skill: it is not a summary. Do not drop facts,
numbers, examples, or personal stories. If the speaker mentioned it, it belongs in the
article.

Beyond faithfulness, this skill has a second goal that shapes everything below: the article
should be a **rich, unhurried read that gives the learner lots of exposure** to the episode's
core words and structures. Learners remember a word after they meet it several times in
context — not once. So the finished article is meaningfully **longer** than a tight retelling
would be (aim for roughly **2–3× the length** of a minimal faithful rewrite), and that extra
length comes from genuine depth and repetition, never from padding.

---

## Core principles

1. **Completeness *and depth* over brevity.** Walk through the transcript in order and make
   sure every idea survives: every statistic (rankings, percentages, yen amounts), every
   country or place name, every personal anecdote, every example the speaker gives. A reader
   who only read your article should learn everything a listener would. Then go one step
   further: **unpack** each point instead of stating it once and moving on. Spell out the
   reasoning the speaker implies, walk through an example step by step, restate a key claim a
   second way to make it land, and add the natural connective explanation that spoken delivery
   leaves implicit. This is how the article reaches 2–3× length **honestly** — you are
   drawing out what is already in the transcript, not inventing new facts. If a fact isn't in
   the transcript, don't add it; deepen what *is* there.

2. **Recycle key vocabulary and grammar so it sticks.** Identify the handful of words and
   expressions that are central to the episode (its topic nouns, the verbs the speaker leans
   on, the grammar patterns you'll list at the end). Then make sure the learner **meets each
   of them several times across the article, in different sentences and contexts** — in the
   はじめに, again in the body where it does real work, and once more near the 結び. Vary the
   surrounding sentence so it feels natural rather than copy-pasted, but keep the target word
   itself intact so the learner recognizes it on each encounter. This repeated, in-context
   exposure is the main reason the article is worth reading over a bare summary, and it's the
   primary way the length grows. Repetition should feel like a good teacher circling back to
   an important word, not like filler.

3. **Rewrite the voice, but stay at the transcript's vocabulary level.** Convert casual
   spoken Japanese into natural written Japanese: smooth out filler, false starts, and
   accidental repetition (as opposed to the *deliberate* recycling above), and merge
   fragmented spoken lines into complete sentences. Prefer 丁寧語 (です・ます). You may tidy
   very casual wording into cleaner written equivalents (e.g. 「むちゃむちゃ低い」→「かなり低い」,
   「やばくない?」→「驚くべきことに」). But **do not raise the difficulty** — stick close to
   the words the speaker actually used, and don't swap plain vocabulary for showier or rarer
   synonyms just to sound sophisticated. The learner is studying *this* episode's language, so
   the words on the page should be the episode's own words, seen more times, not a harder
   vocabulary set.

4. **Teach through grammar.** Triple-bold notable grammar patterns and expressions inline in
   the body so learners notice them (see Highlighting below). Because you're recycling key
   patterns across sections, the same pattern may show up several times — highlight it on its
   meaningful occurrences so the learner sees it repeatedly. Those patterns are the raw
   material for the grammar list at the end. Do **not** bold plain vocabulary.

5. **Write for a learner, not an academic.** Clear, flowing paragraphs with an easy,
   patient rhythm. No bullet points in the article body — bullets are only for the
   vocab/grammar section at the end.

---

## Output structure

Always follow this order. Write the whole thing in one Markdown file.

```
# Ep.[number] [Japanese title]

## はじめに
（intro）

## [Body section heading]
### [optional subsection]
（prose … repeat sections to cover the whole transcript）

## 結び：[short closing subtitle]
（closing）

---

## 重要文法 (Grammar)
- **〜pattern**：繁體中文 short meaning
  - ニュアンス：繁體中文 explanation of the nuance / when to use / common confusion
  - ⭕ 正しい：correct Japanese sentence（繁體中文 translation）
  - ❌ 間違い：incorrect Japanese sentence（繁體中文 note on why it's wrong + the fix）
（5–7 items）
```

### 1. Title

`# Ep.[number] [Japanese title]`

Use the episode number and topic from the transcript's first lines/filename. Japanese only —
no English, no furigana. Example: `# Ep.662 日本人のお給料が低い理由`.

If the transcript has no episode number, use a natural Japanese title alone
(`# 日本人のお給料が低い理由`).

### 2. はじめに (Introduction)

A `## はじめに` section that eases the reader in. Include, when the transcript provides them:
the host/channel greeting (e.g. 「みなさん、こんにちは😊Bite Size Japaneseのレイラです！」),
a one-line description of what kind of program it is, the episode's theme, and a couple of
rhetorical questions that hook the reader (mirroring how the speaker opens). If the speaker
plugs a membership/Patreon/channel, keep that mention here. Write 3–5 short paragraphs, and
introduce the episode's key words here for the first time so the learner starts building
familiarity before the main content.

Apply inline bold highlighting here too, starting from the intro.

### 3. Body

The heart of the article. Divide the content into thematic `##` sections that follow the
transcript's flow, using `###` subsections when a section has distinct parts (e.g. a big
topic with several sub-points or a couple of separate anecdotes). Aim for however many
sections the content actually needs — do not force a fixed number, and do not compress
multiple distinct topics into one thin section just to hit a target.

Within each section, write flowing Japanese prose that faithfully **and fully** re-tells that
part of the transcript, including all specifics. Give each point room to breathe: state it,
illustrate it with the speaker's own example, and where it helps comprehension, restate the
core idea a second way. Convert 「私」-narrated anecdotes into clean written form but keep
every concrete detail (amounts, durations, names, sequence of events); tell longer stories
(like a multi-step complaint) step by step rather than collapsing them. This section is where
most of the 2–3× expansion happens — through depth and the deliberate re-use of the episode's
key words and grammar, not through invented content.

### 4. 結び (Closing)

A `## 結び：[short subtitle]` section that wraps up, mirroring the speaker's outro:
caveats/humility the speaker expressed, any positive counterpoints, the final takeaway, and
a brief thank-you/sign-off if present. This is a natural place to bring back the episode's key
words one last time, so the learner meets them again as the article closes.

### 5. Divider then 重要文法

Put a `---` line, then the grammar section (next).

---

## Highlighting inside the article

Mark grammar so learners can pause on it — and **only** grammar:

- **`***…***` (triple asterisk)** — for **grammar patterns and notable expressions** a
  learner should study (e.g. ***べき***, ***づらく***, ***にもかかわらず***, ***わけではない***,
  ***に対して***, ***に比べて***). Wrap the pattern itself where it appears in the sentence.
  Because key patterns recur across the article, highlight them on their meaningful
  reappearances too — repeated highlighting reinforces recognition.

Do **not** bold plain vocabulary with `**…**`, and do **not** add furigana/reading markup in
the body. The body stays clean, with only the grammar patterns highlighted. Highlight
generously — you can triple-bold more occurrences than end up in the final list; the list is
a curated subset.

---

## 重要文法 section (the grammar list)

Header: `## 重要文法 (Grammar)`. Pick **5–7** grammar patterns that appear in the transcript
(the ones you triple-bolded). For each, don't just translate it — explain the nuance and
show a correct vs. incorrect usage so the learner understands *when* it applies. Format:

```markdown
- **〜に対して**：對於～、相對於～
  - ニュアンス：用來點出「對象」或做前後對比，語感偏正式、帶有一種「面對／針對」的方向性。常和帶對立或比較意味的內容搭配；日常「和某人隨意聊天」不適合用它。
  - ⭕ 正しい：物価の上昇に対して、給料はほとんど上がっていない。（相對於物價上漲，薪水幾乎沒漲。）
  - ❌ 間違い：友達に対して映画を見に行った。（想表達「和朋友」時應用「〜と」：友達と映画を見に行った。「〜に対して」帶對立／針對語感，不適合。）
```

Rules:
- Pattern: leading `〜`, bolded, followed by a short 繁體中文 meaning.
- ニュアンス: 2–3 sentences of **Traditional Chinese** explaining the nuance, the register,
  and the mistake learners commonly make — this is the part the user cares most about, so
  make it genuinely informative, not a one-line gloss.
- ⭕ 正しい: a natural, correct Japanese sentence (prefer the transcript's own usage, lightly
  cleaned) with a 繁體中文 translation in （）.
- ❌ 間違い: a plausible *wrong* sentence a learner might produce, with a 繁體中文 note
  explaining why it's wrong and giving the corrected form. Make the error realistic — a real
  confusion point (wrong particle, wrong pattern for the meaning, register mismatch), not a
  random broken sentence.
- Never use Simplified Chinese. Use Taiwan vocabulary/characters (e.g. 資訊, 影片, 網路).

---

## Processing steps

1. Read the entire transcript (and filename) fully before writing.
2. Note the episode number, title/topic, host/channel name, and any membership plug.
3. Pick out the episode's **key vocabulary and grammar** — the handful of words/patterns worth
   recycling — and plan to reintroduce each of them in several sections.
4. Segment the transcript into thematic chunks; each becomes a `##` (or `###`) section.
5. Draft the article top to bottom, rewriting spoken → polished written Japanese, keeping
   **every** fact, number, name, and anecdote, and **expanding each point with depth and
   deliberate re-use of the key words/grammar** so the article runs roughly 2–3× a minimal
   retelling. Triple-bold grammar patterns as you go; do not bold plain vocabulary; do not
   raise the vocabulary difficulty above the transcript's own level.
6. Write the 結び closing, bringing the key words back one last time.
7. Curate 5–7 grammar items from what you triple-bolded; for each write a 繁體中文 nuance
   explanation plus a correct (⭕) and an incorrect (❌) usage example.
8. Verify: (a) skim the transcript again and confirm no major point was dropped —
   completeness is still the top priority, and confirm the added length is real depth/
   repetition rather than invented facts or filler; (b) confirm the key words genuinely recur
   across sections and that the vocabulary stays close to the transcript's level; (c) confirm
   every grammar item genuinely appears in the transcript and that each ❌ example reflects a
   real, plausible learner mistake; (d) confirm all Chinese is Traditional, not Simplified.

## Multiple transcripts

Produce one article per episode/video. If the user uploads several related ones, you may add
a short `## 共通メモ` note at the very end pointing out shared vocabulary or themes.

## Output file

Save as a `.md` file named after the episode (e.g. `Ep.662 日本人のお給料が低い理由.md`) so
it drops straight into a notes app. Do not add furigana markup or an English translation of
the article body — this format keeps the body Japanese-only, highlights only grammar, and
puts learner support in the final 重要文法 section.
