# time-series-forecasting-ai-systems — Quarto course-site rules

This repo publishes a three-day capstone course as a [Quarto](https://quarto.org)
website to GitHub Pages, built the same way as the `agentic-ai-systems` sibling
site in this workspace: a readable lesson (`.qmd`) paired with a runnable lab
(`.ipynb`), rendered directly into the site rather than shipped as a separate
"golden-thread" microservice. There is no custom Python package here and no
Docker service shipped to students — `common/` is two small tested utility
files, nothing more. Read this before adding a lesson, a lab, or touching
`_quarto.yml`.

## Repo layout

- `index.qmd`, `setup.qmd`, `capstone.qmd`, `assessment.qmd` — site root pages.
- `day1/`, `day2/`, `day3/` — each day is a `NN_topic.qmd` lesson immediately
  followed by its `NN_lab_*.ipynb` lab, both rendered straight into the site
  (see `_quarto.yml`'s `render:` list for the exact, current file set — that
  list is the map of what should exist, more current than this prose).
- `reference/` — `metrics_cheatsheet.qmd`, `tooling_guide.qmd`,
  `troubleshooting.qmd`.
- `common/` — `metrics.py` and `backtest.py`, the only shared code in this
  repo, unit-tested by `tests/test_metrics.py` and `tests/test_backtest.py`.
- `data/` — the four generated CSVs plus `generate_series.py`, their source
  of truth.
- `colab-sim/` — a Docker harness that stands in for a bare, fresh Colab
  runtime. Not part of the shipped course; it's how a notebook's own
  `pip install` setup cell gets proven to actually work rather than riding on
  packages already present on the machine that wrote it.

## Render, don't just read source

A `.qmd`/`.ipynb` file looking right in the editor proves nothing — Quarto's
pandoc pass changes markdown structure in ways invisible in source and only
visible in the rendered HTML. Before calling any page done:

```
quarto render
```

then check the actual output under `_site/**/*.html`, or use the preview
server (`quarto preview`, port 4721 per `_quarto.yml`).

## The standard lab notebook setup cell — don't deviate from its shape

Every lab notebook's first code cell installs its own packages, then fetches
`common/metrics.py` (and `common/backtest.py`, for labs that backtest) plus
whatever dataset CSV it needs, via a `fetch()` helper that tries a local repo
checkout first and falls back to `raw.githubusercontent.com/MohammadYusif/
time-series-forecasting-ai-systems/main/...`. That's what lets the identical
notebook run from a cloned repo, inside `colab-sim/`, and fresh on Colab with
nothing cloned. Every notebook also seeds `RNG_SEED = 20260912` — the same
seed `data/generate_series.py` uses — so don't pick a different seed for a
new lab; there's no reason for one lab's randomness to be reproducible
differently from the data it's forecasting. When writing a new lab, copy this
cell's shape from an existing one and change only the package list — not the
`fetch()` logic, not the seed, not the import style.

## Notebook gotchas (both found the hard way on the `modern-data-engineering-ai` sibling site — treat as binding here too)

- **A markdown cell must never start with a bare `---`.** Quarto's
  ipynb-to-document conversion also uses `---` as a cell separator, so a
  leading horizontal rule breaks the render with a `YAMLException` that
  points at unrelated content several cells later. Use a `##` heading
  instead. Check every markdown cell before trusting one, not just whichever
  one an error blamed:
  ```
  python -c "import json; nb=json.load(open('path.ipynb', encoding='utf-8')); [print(i, repr(''.join(c['source'])[:10])) for i,c in enumerate(nb['cells']) if c['cell_type']=='markdown']"
  ```
- **A markdown cell's `source` must be a list of lines, not one flat
  string.** A flat string renders as a single heading tag that silently
  swallows the entire rest of the cell — no error, no warning, just a badly
  broken heading that's easy to miss on a skim. `nbformat.v4.new_markdown_cell(...)`
  already produces the correct list form; just don't hand-edit `source` back
  into a flat string afterward.

## The Colab-badge trap

Every lab notebook's first cell is an "Open in Colab" badge — always
written as raw HTML, never as a markdown linked image on its own line:
Quarto's implicit-figures pass silently drops the enclosing `<a>` when a
linked image is the only thing in its paragraph, so the badge still *looks*
right in preview with its click target gone. Every existing badge cell
already follows this shape; copy one verbatim for a new lab rather than
writing a fresh one by hand:
```html
<a href="https://colab.research.google.com/github/MohammadYusif/time-series-forecasting-ai-systems/blob/main/dayN/NN_lab_x.ipynb" target="_blank" rel="noopener"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```
The badge cell is a plain markdown cell prepended in each `_build/build_labN.py`
(via `nbf.v4.new_markdown_cell(...)` as the very first entry in the cell
list) — it is not something Quarto or GitHub renders automatically, and it
carries no `execution_count` to verify, unlike every code cell after it.

## Every lab notebook ships with real executed output

A notebook can read as complete — cells defined, output further down —
while a whole section was never actually run. Before treating any lab
notebook as done, check every code cell:

```
python -c "import json; nb=json.load(open('path.ipynb', encoding='utf-8')); [print(i, c.get('execution_count'), len(c.get('outputs',[]))) for i,c in enumerate(nb['cells']) if c['cell_type']=='code']"
```

Every code cell needs a non-null `execution_count` (a cell that legitimately
produces no output, e.g. a bare assignment, can show 0 outputs — that's
fine; `None` is not). Then re-run the notebook inside `colab-sim/` — a
container with nothing preinstalled — before trusting it:

```bash
docker build -t tsf-colab-sim -f colab-sim/Dockerfile .
docker run --rm -v "$(pwd)":/work tsf-colab-sim python colab-sim/run_labs.py
```

A notebook that only runs on the machine that authored it may be relying on
a package that happened to already be installed there; a fresh Colab runtime
(or a student's own machine) won't have that shortcut. Clean output in
`colab-sim/out/` (gitignored) is what proves the committed output in
`day1/`/`day2/`/`day3/` is real and current, not hand-written or carried
over from an earlier version of the code.

## The datasets are generated, never hand-edited

`data/generate_series.py` is the source of truth for all four CSVs — it's
seeded (`SEED = 20260912`) and deterministic, so re-running it reproduces
them byte-for-byte. If a dataset ever needs to change (a new shock, a
different noise level), edit the generator and re-run it; never hand-edit a
CSV under `data/` directly. Commit the regenerated CSV alongside the
generator change so the diff shows what actually moved. Every one of these
series is synthetic — no lesson, lab, or reference page should describe a
number from them as if it came from a real retailer, employer, or economic
series.

## `common/` stays small on purpose

`common/metrics.py` and `common/backtest.py` are the only shared code this
repo has, and that's deliberate — this course is intentionally built lighter
than `llm-application-engineering`'s `murshid/` golden-thread project, with
no shared package to maintain beyond these two files. Both are unit-tested
(`tests/test_metrics.py`, `tests/test_backtest.py` — run with `pytest`).
A genuine need specific to one lab (a helper only `03_lab_lightgbm_forecast.ipynb`
uses, say) stays inline in that notebook rather than growing a third shared
file; only add to `common/` when at least two labs would otherwise duplicate
the exact same logic, the way `metrics.py`'s docstring already explains its
own reason for existing.

## `_quarto.yml` gotchas

- **`execute: enabled: false` project-wide is deliberate.** Lab notebooks are
  committed with real executed output (produced locally and re-verified in
  `colab-sim/`); Quarto renders those saved outputs and never re-executes
  anything, so the site builds in CI with no dependency install and no
  compute. Don't flip this on for the whole project to "make sure it's
  fresh" — re-execute and re-verify the specific notebook instead (see
  above), then re-render.
- **`data/`, `common/`, `tests/`, and `colab-sim/` are excluded from the
  render list** (`"!data/"` etc.) because none of them are pages. A new
  top-level folder that isn't meant to be a site section needs the same
  exclusion, or `quarto render` will try to render whatever's in it.
- **Don't add `revealjs` as a project-level format.** It makes Quarto render
  every page twice, both claiming the same output filename, and the whole
  project render dies. If a page ever needs slides, render that one file
  explicitly (`quarto render <page>.qmd --to revealjs`) with the slide
  options in its own front matter.
- Keep the sidebar's `contents:` list in `_quarto.yml` in sync with what
  actually exists under `day1/`/`day2/`/`day3/`/`reference/` — a page not
  listed there won't show in navigation even if it renders fine standalone.

## No API key, minimal network

Nothing in this repository needs an API key, a GPU, or standing network
access. The only network calls a lab notebook makes are the one-time
`pip install`s for its own packages and the `fetch()` helper's download of
`common/*.py` and the dataset CSV it needs — and both of those no-op (or
resolve to a local file) when the notebook is already running from a repo
checkout, including inside `colab-sim/`. If something here seems to need a
`.env` or a secret, that's a sign the setup cell was written wrong, not that
one is missing.

## General rule

Before calling a lesson or a lab "done": re-render the site and check
`_site/`, and for a notebook, re-check its `execution_count`/`outputs` and
re-run it inside `colab-sim/` — not just that the source reads correctly.
