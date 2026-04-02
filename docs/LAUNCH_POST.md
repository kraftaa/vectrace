# Launch Post Draft

## Title

LLM gave a wrong answer? I built a CLI to trace exactly where it came from.

## Body

Most RAG debugging starts in the wrong place.

People tweak prompts or switch models before answering one basic question:
**which source chunk actually caused this answer?**

So I built **VecTrace**.

`vectrace` is a small Python CLI that traces a retrieved vector back to:
- source document
- chunk text
- embedding model
- pipeline run metadata

### A concrete failure case

- User asks: `Can I get a refund after 90 days?`
- Assistant says: `Yes, refunds are allowed.`
- Team expectation: refunds are only valid for 30 days.

Run:

```bash
vectrace trace --vector-id vec_101 --collection support_kb --db ./vectrace.db
```

It points to:
- `s3://kb/refund_policy_old.md` (outdated file)
- a chunk that still says 90-day refunds

Now you know the fix: retrieval/index data, not prompt hacks.

### What VecTrace does

- Track trace records (`document -> chunk -> vector`)
- Trace vector IDs to source
- Generate a shareable HTML report (`vectrace report`)
- Output JSON for CI (`--format json`)

Install:

```bash
pip install vectrace
```

30-second demo:

```bash
vectrace onboard --db ./vectrace.db --output ./trace.html
vectrace trace --vector-id vec_demo_001 --collection support_kb --db ./vectrace.db
vectrace report --vector-id vec_demo_001 --collection support_kb --db ./vectrace.db --output ./trace.html
```

If you use RAGLens, this fits right after retrieval inspection:

```bash
raglens explain --query "Can I get a refund after 90 days?" --top-k 5
vectrace trace --vector-id vec_101 --collection support_kb --db ./vectrace.db
```

Repo: `https://github.com/your-org/vectrace`

VecTrace is one piece of a bigger goal: **observability for AI pipelines**.

## Short Version (X/LinkedIn)

Wrong RAG answer? Don’t guess.

Use **VecTrace** to trace the exact source:
`vector -> chunk -> document`

```bash
pip install vectrace
vectrace trace --vector-id vec_101 --collection support_kb --db ./vectrace.db
```

Shareable HTML report + JSON mode for CI.

Repo: https://github.com/your-org/vectrace
