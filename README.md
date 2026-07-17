# unbugify

**Score a Python codebase and work the queue until it is clean.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Dependencies: none](https://img.shields.io/badge/dependencies-0-brightgreen)

Most linters hand you a wall of two thousand warnings and leave. You look at it
once, decide today is not the day, and never look again.

unbugify gives you one number, a queue ordered worst-first, and a memory. It
remembers what it told you last time, so every scan can say *you fixed nine, two
came back, here is the next one*. The number only moves when the code actually
gets better — it is designed to resist the usual tricks.

Zero runtime dependencies. Python 3.11+. Pure stdlib `ast`.

---

## Install

```bash
pip install unbugify
```

Or just take the file. `unbugify.py` has no dependencies beyond the standard
library, so vendoring it into a project that cannot take a new dependency is a
copy and nothing else:

```bash
curl -O https://raw.githubusercontent.com/anticam8-pixel/Unbugify/main/unbugify.py
python unbugify.py scan
```

## Use

```bash
unbugify scan               # score the tree, show the worst offenders
unbugify next               # the next three things worth fixing
unbugify scan               # ... and see what moved
```

Add `.unbugify/` to your `.gitignore` — it holds local state.

```
  89.7/100  B   Solid, with pockets of mess.
  strict 89.7 | 1,934 sloc | 17 findings

  +6.3 since last scan
  9 fixed · 2 new

  By severity
    high      3
    medium    6
    low       49

  Worst first

  1. [high] core/parser.py:57-88  cognitive
     parse_header() has cognitive complexity 27 (limit 15) -- hard to hold in your head.
     id 7d67176d6ff1
```

### Commands

| Command | What it does |
| --- | --- |
| `unbugify scan` | Scan, score, record. `--format json\|markdown`, `--fail-under N`, `--dry-run` |
| `unbugify next` | Show the next N findings from the queue (`--count`) |
| `unbugify report` | Full report to stdout or `--output FILE` |
| `unbugify badge` | Self-contained SVG score badge for your README |
| `unbugify history` | Score over time |
| `unbugify ignore <id>` | Mute a finding by its id |
| `unbugify unignore <id>` | Unmute it |
| `unbugify exclude <path>` | Exclude a directory from scans |
| `unbugify checks` | List the checks |

## What it looks for

| Check | Finds |
| --- | --- |
| `complexity` | Cyclomatic and cognitive complexity, nesting depth, function/module length, parameter counts |
| `deadcode` | Unused imports, unreferenced private symbols, unreachable statements, orphaned public API |
| `duplication` | Copy-pasted statement runs, within and across modules |
| `smells` | Bare/broad excepts, silently swallowed exceptions, mutable default arguments, wildcard imports, `global`, `== None`, debt markers |
| `docs` | Missing docstrings on the non-trivial public surface |

Two details worth knowing, because they are where most tools get it wrong:

**`elif` is not nesting.** Python has no elif node — an `elif` is an `If` living
inside the parent's `orelse`. Naive tools count a flat five-branch dispatch chain
as five levels deep. unbugify does not.

**Duplication works on statements, not lines.** A call spread across eight lines
in your house style is one statement, not eight duplicated lines. Line-based
detectors light up like a christmas tree on consistently-formatted code;
unbugify normalizes literals away, hashes runs of consecutive AST statements, and
only reports real copies.

## The score

Start at 100. Every finding carries a weight by severity (critical 10, high 5,
medium 2, low 1). Sum the weights, divide by thousands of source lines, and run
the result through an exponential decay curve.

```
score = 100 · e^(−density / 95)      density = Σ weights / KLOC
```

Three properties this buys you:

1. **You cannot game it by deleting code.** Penalties are a *density*. Delete the
   parts that work and the remaining rot gets more concentrated, not less. There
   is a test for exactly this.
2. **Muting is not winning.** `unbugify ignore` drops a finding to 25% weight in
   the headline score and leaves it at full weight in the **strict** score, which
   is printed right next to it. You can silence something you have genuinely
   accepted; you cannot silence your way to an A.
3. **No cliffs to sit under.** The curve is smooth and asymptotic. There is no
   threshold to game, 100 means nothing was found, and the last few points are
   the expensive ones — which is how real cleanup actually feels.

| Score | Grade |
| --- | --- |
| 98+ | A+ — beautiful |
| 93+ | A — clean |
| 85+ | B — solid, with pockets of mess |
| 75+ | C — workable, the debt is visible |
| 60+ | D — cleanup is now cheaper than the alternative |
| <60 | F — this is costing you real time every week |

## Configure

In `pyproject.toml`:

```toml
[tool.unbugify]
exclude = ["vendor", "generated"]
include_tests = false
disable = ["docs"]

[tool.unbugify.thresholds]
max_cyclomatic = 10
max_cognitive = 15
max_function_lines = 60
max_module_lines = 500
max_parameters = 5
max_nesting = 4
min_duplicate_statements = 5
```

## CI

```yaml
- name: Code quality gate
  run: |
    pip install unbugify
    unbugify scan --fail-under 80 --no-color
```

Ratchet the number up over time. It is very hard to argue with a gate that only
ever moves in one direction.

## Use it with a coding agent

The scan output is designed to be pasted straight into an agent. Try:

```
Run `unbugify scan` on this repo, then `unbugify next`. Fix the top finding,
re-scan to confirm the score moved, and repeat. Do not mute anything without
asking me first.
```

`--format json` gives structured findings with stable ids if you want to drive
it programmatically.

## Design notes

- **Fingerprints ignore line numbers.** A finding is identified by check, rule,
  file, and a content snippet — so adding an import at the top of a file does not
  make every finding below it look "fixed and re-found".
- **Cross-file symbol usage.** Dead-code detection considers the whole scanned
  tree, so a helper used in another module is not reported as dead.
- **Deterministic output.** Two scans of the same tree produce byte-identical
  JSON. Tested.
- **Conservative on public API.** An unreferenced public function is reported at
  `low` with a warning to check external callers first, never at `high`.

## Layout

The whole tool is one module. That is a deliberate trade: it makes unbugify
trivial to vendor into a project that cannot take a dependency, at the cost of a
file far longer than its own `module-too-long` rule recommends.

It flags itself for exactly that, and it is right to. The finding is left in
rather than muted or threshold-tuned away, because a quality tool that quietly
exempts itself from its own rules is not worth running. That single `critical`
costs about six points:

```
  84.2/100  C   Workable, but the debt is visible.
  strict 84.2 | 1,834 sloc | 18 findings

  [critical] unbugify.py:1  module-too-long
     Module is 2261 lines (limit 500).
```

Inside the file, the original module boundaries survive as banner comments:
models, config, discovery, the five checks, scoring, state, reporting, badge,
engine, CLI -- in dependency order, nothing referencing a name defined below it.

## Develop

```bash
pip install -e ".[dev]"
pytest
python unbugify.py scan          # it scans itself
```

## License

MIT — see [LICENSE](LICENSE).
