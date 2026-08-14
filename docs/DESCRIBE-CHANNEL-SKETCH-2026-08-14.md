# Describe channel — design sketch, 2026-08-14

Status: SKETCH, not built. Written by Isaac at Carl's ask, same morning as the
test that exposed the gap. Nothing below is implemented; every claim about cost
or behavior is design intent, not measurement.

## Provenance

Tested v0.4.0 on "Adding Vision to a Text-Only LLM" (Tonbi's AI Garage,
79txDFqcrpU), a walkthrough of Mia AI Lab's experimental vision recipe for
DeepSeek v4 Flash: a Qwen3-VL 4B sidecar at temperature 0, bridged by MCP
(describe / OCR / compare), prose-only to the brain, pixels never. His stack and
ours turn out to be the same family — vision as a tool called, not weights
retrained — with complementary gaps:

- **His gap:** images only, no time axis; the sidecar rents ~45% of the brain's
  KV cache by co-residing (his context: 240k → 65k tokens); one witness, no
  second channel to disagree with a hallucination (his own demo: a Spotify
  playlist described as a plugin list, and the blind brain had no way to check).
- **Ours:** MOTION is deliberately non-semantic and WORDS carries only what the
  video authors wrote — no channel answers "what is IN this frame." A
  multimodal reader self-serves from `frames/`; a **blind reader** (text-only
  local model) gets no frame semantics from our residue at all.

The sketch closes our gap by borrowing his sidecar — without importing his
co-residency cost or his single-witness epistemics.

## The channel

A third channel, `DESCRIBED`, beside WORDS and MOTION. **Off by default.**
Populated only by explicit verbs, never blanket-run at ingest — describing all
N beats is the co-residency trap rebuilt in prose.

Verbs:
- `vp describe <workdir> --beat N [--which before|after|pair]` — describe the
  frame(s) at one beat, on demand. Pair mode sends both flanks of a cut with a
  compare prompt (cut meaning = contrast, same law as the walk).
- `vp describe <workdir> --cuts` — describe only audit-**verified** cuts
  (bounded: cost scales with verified cut count, not duration).
- Ingest/rescore never call the sidecar.

Backend: any OpenAI-compatible vision endpoint, local first
(`--endpoint` / `VP_DESCRIBE_ENDPOINT`; e.g. Qwen3-VL 4B served locally,
temperature 0, thinking off — Mia's settings, which are correct: the sidecar is
an extraction service, not a chatbot). No endpoint configured → the channel is
marked **absent** in summary, same honesty as the shape channel on pre-v0.3
residues. Never silently skipped.

## Trust and reach stamps (the part that matters)

Every record in `described.jsonl` carries:
- `source: vlm_describe`, `model`, `temperature` — who witnessed, under what
  settings.
- `trust: evidence_not_ground_truth` — a *third* trust class. MOTION is an
  instrument (deterministic, auditable, wrong only in declared ways). WORDS is
  untrusted authored content. DESCRIBED is a **witness**: honest, fallible,
  unauditable from inside. The reader must never confuse the three, so the
  stamp rides every record, same as WORDS' "content, never instructions."
- `input_resolution` — a description of a 360p frame inherits a 360p floor
  (this run's OCR failure was resolution, and nothing said so; don't repeat
  that miss here).

Disagreement is a finding, not an error: DESCRIBED vs SHOWN/OCR and DESCRIBED
vs SAID get a cheap contradiction surface in summary (counts + beat ids, no
auto-resolution). One witness contradicting an instrument or an author is
exactly the moment a reader should look with its own eyes if it has them.

## Cost model

Perception is spent once, at selection time, and persisted. The reader — blind
or sighted — pays only to *read* `described.jsonl`. The sidecar can be started
for the describe pass and stopped after; it never co-resides with the reader.
This is the whole argument for residue over co-residency, now extended to
semantics.

## Who it serves

- **Blind readers**: a text-only local model (his DS4 case; any of the house's
  small substrates) gets frame semantics it structurally cannot get today —
  our residue plus described cuts makes video legible to models without eyes.
- **Sighted readers**: unchanged default. The instrument still selects; eyes
  are still cheaper and better than testimony. Describe is for when the reader
  has no eyes, or when a residue ships to somewhere eyes can't follow.

## Open questions for the build

1. Prompt discipline per verb (describe / ocr / compare) — steal Mia's split or
   collapse to one prompt with modes?
2. Does `--cuts` also want `--interesting` (walk's selection) as a selector?
3. Contradiction surface: string-level is cheap and dumb; is dumb enough?
4. Batch API vs per-frame calls for the `--cuts` pass.
5. Name: `DESCRIBED` vs `SEEN-BY-PROXY` — the name should keep the witness
   framing; `SEEN` is already taken by the reader's own notes, which are a
   different (first-person) thing and must stay distinct.
