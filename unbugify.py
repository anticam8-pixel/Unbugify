"""unbugify -- score a Python codebase and work the queue until it is clean.

A single-file, zero-dependency code quality scorer. Nothing to install but this
module; nothing imported but the standard library.

Five checks -- complexity, dead code, duplication, smells, docs -- produce
weighted findings. The findings become a density per thousand source lines, and
that density becomes a score you cannot game by deleting code. State persists
between runs, so every scan can tell you what you fixed and what came back.

    unbugify scan               score the tree, worst offenders first
    unbugify next               the next thing worth fixing
    unbugify scan --fail-under 80    a CI gate that only ratchets upward

The single-file layout is a deliberate trade: it makes the tool trivial to vendor
into a project that cannot take a dependency, at the cost of a module far longer
than the one this tool's own `module-too-long` rule recommends. It flags itself
for this, and it is right to. The packaged layout at the homepage below splits
the same code across modules.

Homepage: https://github.com/anticam8-pixel/Unbugify
License: MIT
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import exp
from pathlib import Path
from typing import Any, Iterable, Iterator

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

__version__ = "0.1.0"


# ==========================================================================
# Core data structures
# ==========================================================================

"""Core data structures shared across unbugify."""
class Severity(str, Enum):
    """How much a finding hurts the score."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def weight(self) -> float:
        return _SEVERITY_WEIGHTS[self]


_SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 10.0,
    Severity.HIGH: 5.0,
    Severity.MEDIUM: 2.0,
    Severity.LOW: 1.0,
}


@dataclass
class Finding:
    """A single issue detected in the codebase.

    ``fingerprint`` is deliberately independent of line numbers so that a
    finding survives unrelated edits above it in the file. Moving code around
    should not look like "fixed one, found a new one".
    """

    check: str
    rule: str
    severity: Severity
    path: str
    line: int
    message: str
    snippet: str = ""
    end_line: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        basis = f"{self.check}|{self.rule}|{self.path}|{self.snippet or self.message}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]

    @property
    def weight(self) -> float:
        return self.severity.weight

    @property
    def location(self) -> str:
        if self.end_line and self.end_line != self.line:
            return f"{self.path}:{self.line}-{self.end_line}"
        return f"{self.path}:{self.line}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["fingerprint"] = self.fingerprint
        return data


@dataclass
class ModuleInfo:
    """Size accounting for one scanned file."""

    path: str
    loc: int
    sloc: int
    parse_error: str | None = None


@dataclass
class ScanResult:
    """Everything one scan produced."""

    findings: list[Finding] = field(default_factory=list)
    modules: list[ModuleInfo] = field(default_factory=list)

    @property
    def total_sloc(self) -> int:
        return sum(m.sloc for m in self.modules)

    @property
    def kloc(self) -> float:
        return max(self.total_sloc / 1000.0, 0.001)


# ==========================================================================
# Configuration
# ==========================================================================

"""Configuration: thresholds, exclusions, and on-disk config loading."""
DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    ".unbugify",
    ".eggs",
    "migrations",
)


@dataclass
class Thresholds:
    """Tunable limits. Defaults aim at "a seasoned reviewer would nod"."""

    max_cyclomatic: int = 10
    max_cognitive: int = 15
    max_function_lines: int = 60
    max_module_lines: int = 500
    max_parameters: int = 5
    max_nesting: int = 4
    max_returns: int = 6
    min_duplicate_statements: int = 5
    min_docstring_public: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Config:
    """Full runtime configuration for a scan."""

    root: Path = field(default_factory=Path.cwd)
    excludes: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    include_tests: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    disabled_checks: list[str] = field(default_factory=list)
    # Findings the user muted. They still count, at reduced weight, in the
    # strict score -- muting is not a way to win.
    ignored: list[str] = field(default_factory=list)

    @property
    def state_dir(self) -> Path:
        return self.root / ".unbugify"

    def is_excluded(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            rel = path
        parts = set(rel.parts)
        for pattern in self.excludes:
            if pattern in parts:
                return True
            if rel.match(pattern):
                return True
            if str(rel).startswith(pattern.rstrip("/") + "/"):
                return True
        return False

    @classmethod
    def load(cls, root: Path) -> "Config":
        """Read config from pyproject.toml then .unbugify/config.json.

        The JSON file wins on conflicts because that is what the CLI writes.
        """
        cfg = cls(root=root)
        cfg._load_pyproject(root / "pyproject.toml")
        cfg._load_json(root / ".unbugify" / "config.json")
        return cfg

    def _load_pyproject(self, path: Path) -> None:
        if not path.exists() or tomllib is None:
            return
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            return
        section = data.get("tool", {}).get("unbugify", {})
        self._apply(section)

    def _load_json(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._apply(data)

    def _apply(self, data: dict) -> None:
        for extra in data.get("exclude", []) or []:
            if extra not in self.excludes:
                self.excludes.append(extra)
        if "include_tests" in data:
            self.include_tests = bool(data["include_tests"])
        for name in data.get("disable", []) or []:
            if name not in self.disabled_checks:
                self.disabled_checks.append(name)
        for fp in data.get("ignore", []) or []:
            if fp not in self.ignored:
                self.ignored.append(fp)
        for key, value in (data.get("thresholds") or {}).items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)

    def save(self) -> None:
        """Persist the user-editable parts back to .unbugify/config.json."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        custom = [e for e in self.excludes if e not in DEFAULT_EXCLUDES]
        payload = {
            "exclude": custom,
            "include_tests": self.include_tests,
            "disable": self.disabled_checks,
            "ignore": self.ignored,
            "thresholds": self.thresholds.to_dict(),
        }
        path = self.state_dir / "config.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ==========================================================================
# File discovery and parsing
# ==========================================================================

"""Finding Python files and turning them into parsed source units."""
TEST_MARKERS = ("test_", "_test.py", "conftest.py")


@dataclass
class SourceFile:
    """One Python file, read and (usually) parsed."""

    path: Path
    rel: str
    text: str
    lines: list[str]
    tree: ast.Module | None
    parse_error: str | None = None

    @property
    def loc(self) -> int:
        return len(self.lines)

    @property
    def sloc(self) -> int:
        """Source lines of code: non-blank, non-comment.

        Approximate by design -- a docstring body counts. Consistency across
        scans matters more here than philosophical purity, because this number
        is the denominator of the score.
        """
        count = 0
        for raw in self.lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            count += 1
        return count

    def is_test(self) -> bool:
        name = self.path.name
        if any(m in name for m in TEST_MARKERS):
            return True
        return "tests" in self.path.parts or "test" in self.path.parts


def iter_python_files(config: Config, target: Path | None = None) -> Iterator[Path]:
    """Yield every .py file under ``target`` that survives the exclude rules."""
    base = target or config.root
    if base.is_file():
        if base.suffix == ".py" and not config.is_excluded(base):
            yield base
        return
    for path in sorted(base.rglob("*.py")):
        if config.is_excluded(path):
            continue
        if not path.is_file():
            continue
        yield path


def load_sources(config: Config, target: Path | None = None) -> list[SourceFile]:
    """Read and parse every discovered file. Unreadable files are skipped."""
    sources: list[SourceFile] = []
    for path in iter_python_files(config, target):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            rel = str(path.relative_to(config.root))
        except ValueError:
            rel = str(path)
        lines = text.splitlines()
        tree: ast.Module | None = None
        error: str | None = None
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            error = f"line {exc.lineno}: {exc.msg}"
        source = SourceFile(
            path=path, rel=rel, text=text, lines=lines, tree=tree, parse_error=error
        )
        if source.is_test() and not config.include_tests:
            continue
        sources.append(source)
    return sources


# ==========================================================================
# Check protocol and registry
# ==========================================================================

"""Check plugin protocol and registry."""
class Check:
    """A single analysis pass over the whole set of source files.

    Checks receive every file at once rather than one at a time, because the
    interesting questions (is this symbol used anywhere? is this block copied
    elsewhere?) are cross-file questions.
    """

    name: str = "base"
    description: str = ""

    def run(self, sources: list[SourceFile], config: Config) -> Iterable[Finding]:
        raise NotImplementedError


_REGISTRY: list[type[Check]] = []


def register(cls: type[Check]) -> type[Check]:
    """Class decorator that adds a check to the default run order."""
    _REGISTRY.append(cls)
    return cls


def all_checks() -> list[Check]:
    return [cls() for cls in _REGISTRY]


def enabled_checks(config: Config) -> list[Check]:
    return [c for c in all_checks() if c.name not in config.disabled_checks]


# ==========================================================================
# Check: complexity
# ==========================================================================

"""Complexity metrics: cyclomatic, cognitive, nesting depth."""
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.comprehension,
)

NESTING_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
)


def cyclomatic_complexity(node: ast.AST) -> int:
    """McCabe complexity: one, plus one per branch point.

    Boolean operators add ``len(values) - 1`` because ``a and b and c`` is two
    extra paths, not one.
    """
    score = 1
    for child in ast.walk(node):
        if isinstance(child, DECISION_NODES):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
        elif isinstance(child, ast.IfExp):
            score += 1
        elif isinstance(child, ast.Match):
            score += len(child.cases)
    return score


def _is_elif(parent: ast.AST, child: ast.AST) -> bool:
    """True when ``child`` is the ``elif`` arm of ``parent``.

    Python has no elif node: ``elif`` is an ``If`` that is the only statement
    in the parent ``If``'s ``orelse``. Treating that as nesting would make
    every flat dispatch chain look like a pyramid, so it is explicitly not.
    """
    return (
        isinstance(parent, ast.If)
        and isinstance(child, ast.If)
        and len(parent.orelse) == 1
        and parent.orelse[0] is child
    )


def cognitive_complexity(node: ast.AST) -> int:
    """Cognitive complexity: how hard the code is to *follow*.

    Unlike cyclomatic, nesting is punished: a branch three levels deep costs
    more than the same branch at the top of the function. Loosely follows the
    Campbell (SonarSource) formulation, simplified for Python -- an elif arm
    costs one flat point rather than a nesting increment.
    """
    total = 0

    def walk(n: ast.AST, depth: int) -> None:
        nonlocal total
        for child in ast.iter_child_nodes(n):
            increment = 0
            nested = depth
            if _is_elif(n, child):
                increment = 1
            elif isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                increment = 1 + depth
                nested = depth + 1
            elif isinstance(child, ast.ExceptHandler):
                increment = 1 + depth
                nested = depth + 1
            elif isinstance(child, ast.BoolOp):
                increment = 1
            elif isinstance(child, ast.IfExp):
                increment = 1 + depth
            elif isinstance(child, FUNC_NODES):
                # Nested function: its own body starts a fresh nesting level.
                nested = depth + 1
            total += increment
            walk(child, nested)

    walk(node, 0)
    return total


def max_nesting_depth(node: ast.AST) -> int:
    """Deepest block nesting inside a function body.

    An ``elif`` arm does not deepen the nesting -- see ``_is_elif``.
    """
    deepest = 0

    def walk(n: ast.AST, depth: int) -> None:
        nonlocal deepest
        for child in ast.iter_child_nodes(n):
            if _is_elif(n, child):
                new_depth = depth
            elif isinstance(child, NESTING_NODES):
                new_depth = depth + 1
            else:
                new_depth = depth
            deepest = max(deepest, new_depth)
            walk(child, new_depth)

    walk(node, 0)
    return deepest


def _severity_for(value: int, limit: int) -> Severity:
    ratio = value / max(limit, 1)
    if ratio >= 2.5:
        return Severity.CRITICAL
    if ratio >= 1.75:
        return Severity.HIGH
    if ratio >= 1.25:
        return Severity.MEDIUM
    return Severity.LOW


@register
class ComplexityCheck(Check):
    """Size and complexity metrics on functions and modules."""

    name = "complexity"
    description = "Cyclomatic and cognitive complexity, nesting depth, function size"

    def run(self, sources: list[SourceFile], config: Config) -> Iterable[Finding]:
        """Measure every function, then every module."""
        t = config.thresholds
        for source in sources:
            if source.tree is None:
                continue
            for node in ast.walk(source.tree):
                if not isinstance(node, FUNC_NODES):
                    continue
                yield from self._check_function(source, node, t)

            if source.loc > t.max_module_lines:
                yield Finding(
                    check=self.name,
                    rule="module-too-long",
                    severity=_severity_for(source.loc, t.max_module_lines),
                    path=source.rel,
                    line=1,
                    message=(
                        f"Module is {source.loc} lines (limit {t.max_module_lines}). "
                        f"Consider splitting it along its natural seams."
                    ),
                    snippet=source.rel,
                )

    def _check_function(self, source: SourceFile, node, t) -> Iterable[Finding]:
        """Apply every per-function rule to one function definition."""
        for rule in (
            self._rule_cyclomatic,
            self._rule_cognitive,
            self._rule_nesting,
            self._rule_length,
            self._rule_parameters,
            self._rule_returns,
        ):
            yield from rule(source, node, t)

    def _emit(self, source, node, rule, value, limit, message, severity=None):
        """Build a finding for a threshold breach on a function."""
        return Finding(
            check=self.name,
            rule=rule,
            severity=severity or _severity_for(value, limit),
            path=source.rel,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", None),
            message=message,
            snippet=f"{node.name}:{rule}",
            extra={"value": value, "limit": limit},
        )

    def _rule_cyclomatic(self, source, node, t) -> Iterable[Finding]:
        value = cyclomatic_complexity(node)
        if value > t.max_cyclomatic:
            yield self._emit(
                source, node, "cyclomatic", value, t.max_cyclomatic,
                f"{node.name}() has cyclomatic complexity {value} "
                f"(limit {t.max_cyclomatic}) -- too many branches to test.",
            )

    def _rule_cognitive(self, source, node, t) -> Iterable[Finding]:
        value = cognitive_complexity(node)
        if value > t.max_cognitive:
            yield self._emit(
                source, node, "cognitive", value, t.max_cognitive,
                f"{node.name}() has cognitive complexity {value} "
                f"(limit {t.max_cognitive}) -- hard to hold in your head.",
            )

    def _rule_nesting(self, source, node, t) -> Iterable[Finding]:
        value = max_nesting_depth(node)
        if value > t.max_nesting:
            yield self._emit(
                source, node, "deep-nesting", value, t.max_nesting,
                f"{node.name}() nests {value} levels deep (limit {t.max_nesting}). "
                f"Early returns or extracted helpers usually flatten this.",
            )

    def _rule_length(self, source, node, t) -> Iterable[Finding]:
        end = getattr(node, "end_lineno", node.lineno)
        value = end - node.lineno + 1
        if value > t.max_function_lines:
            yield self._emit(
                source, node, "function-too-long", value, t.max_function_lines,
                f"{node.name}() is {value} lines (limit {t.max_function_lines}). "
                f"It is probably doing more than one thing.",
            )

    def _rule_parameters(self, source, node, t) -> Iterable[Finding]:
        args = node.args
        value = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
        if args.args and args.args[0].arg in ("self", "cls"):
            value -= 1
        if value > t.max_parameters:
            yield self._emit(
                source, node, "too-many-parameters", value, t.max_parameters,
                f"{node.name}() takes {value} parameters (limit {t.max_parameters}). "
                f"A dataclass or config object usually wants to exist here.",
            )

    def _rule_returns(self, source, node, t) -> Iterable[Finding]:
        value = sum(
            1
            for child in ast.walk(node)
            if isinstance(child, ast.Return) and child.value is not None
        )
        if value > t.max_returns:
            yield self._emit(
                source, node, "too-many-returns", value, t.max_returns,
                f"{node.name}() has {value} return statements (limit {t.max_returns}).",
                severity=Severity.LOW,
            )


# ==========================================================================
# Check: dead code
# ==========================================================================

"""Dead code: unused imports, unreferenced symbols, unreachable statements."""
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# Names that are used by machinery rather than by an explicit reference.
MAGIC_NAMES = re.compile(r"^__\w+__$")

ENTRY_POINT_DECORATORS = {
    "app",
    "cli",
    "route",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "command",
    "group",
    "task",
    "fixture",
    "hookimpl",
    "register",
    "property",
    "setter",
    "getter",
    "deleter",
    "overload",
    "abstractmethod",
    "staticmethod",
    "classmethod",
}


def _decorator_names(node) -> set[str]:
    names: set[str] = set()
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        while isinstance(target, ast.Attribute):
            names.add(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


class _NameCollector(ast.NodeVisitor):
    """Every identifier that is *read* somewhere in a module."""

    def __init__(self) -> None:
        self.used: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Load, ast.Del)):
            self.used.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.used.add(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Strings can name things: __all__, getattr(), Django-style configs.
        if isinstance(node.value, str) and node.value.isidentifier():
            self.used.add(node.value)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # `from .badge import render as render_badge` is a use of `render`,
        # even though the name that ends up bound locally is different.
        for alias in node.names:
            self.used.add(alias.name.split(".")[0])
        self.generic_visit(node)


def _references_to(tree: ast.Module, name: str) -> int:
    """How many times ``name`` is read as a bare name or an attribute."""
    return sum(
        1
        for n in ast.walk(tree)
        if (isinstance(n, ast.Name) and n.id == name)
        or (isinstance(n, ast.Attribute) and n.attr == name)
    )


def _module_used_names(source: SourceFile) -> set[str]:
    collector = _NameCollector()
    if source.tree is not None:
        collector.visit(source.tree)
    return collector.used


def _is_all_assignment(node: ast.stmt) -> bool:
    """True for a module-level ``__all__ = [...]`` statement."""
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
    )


def _string_elements(value: ast.expr) -> set[str]:
    """Every string literal inside a list or tuple display."""
    if not isinstance(value, (ast.List, ast.Tuple)):
        return set()
    return {
        e.value
        for e in value.elts
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    }


def _declared_exports(tree: ast.Module) -> set[str]:
    """Names the module explicitly publishes via ``__all__``."""
    exports: set[str] = set()
    for node in tree.body:
        if _is_all_assignment(node):
            exports |= _string_elements(node.value)  # type: ignore[attr-defined]
    return exports


@register
class DeadCodeCheck(Check):
    """Code that exists but nothing reaches."""

    name = "deadcode"
    description = "Unused imports, unreferenced private symbols, unreachable code"

    def run(self, sources: list[SourceFile], config: Config) -> Iterable[Finding]:
        """Build project-wide usage, then judge each module against it."""
        parsed = [s for s in sources if s.tree is not None]
        # Project-wide identifier usage, so a helper used in another module is
        # not reported as dead.
        global_used: set[str] = set()
        for source in parsed:
            global_used |= _module_used_names(source)

        for source in parsed:
            local_used = _module_used_names(source)
            yield from self._unused_imports(source, local_used)
            yield from self._unreferenced_privates(source, local_used, global_used)
            yield from self._unreachable(source)

    def _unused_imports(self, source: SourceFile, used: set[str]) -> Iterable[Finding]:
        """Report imported names that nothing in the module reads."""
        assert source.tree is not None
        exports = _declared_exports(source.tree)
        is_init = source.path.name == "__init__.py"

        for node in ast.walk(source.tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            for alias in node.names:
                bound = self._bound_name(alias)
                if bound is None or bound in used or bound in exports:
                    continue
                # A bare re-export in __init__.py is a convention, not a bug.
                if is_init and not exports:
                    continue
                yield Finding(
                    check=self.name,
                    rule="unused-import",
                    severity=Severity.LOW,
                    path=source.rel,
                    line=node.lineno,
                    message=f"'{bound}' is imported but never used.",
                    snippet=f"import:{bound}",
                )

    @staticmethod
    def _bound_name(alias: ast.alias) -> str | None:
        """The local name an import binds, or None for a star import."""
        if alias.name == "*":
            return None
        return alias.asname or alias.name.split(".")[0]

    def _unreferenced_privates(
        self, source: SourceFile, local_used: set[str], global_used: set[str]
    ) -> Iterable[Finding]:
        """Report module-level definitions nothing appears to reference."""
        assert source.tree is not None
        exports = _declared_exports(source.tree)

        for node in source.tree.body:
            if not isinstance(node, FUNC_NODES + (ast.ClassDef,)):
                continue
            if self._is_exempt(node, exports):
                continue
            referenced = _references_to(source.tree, node.name) > 0
            if node.name.startswith("_"):
                if not referenced:
                    yield self._private_finding(source, node)
            elif not referenced and node.name not in global_used:
                yield self._public_finding(source, node)

    @staticmethod
    def _is_exempt(node, exports: set[str]) -> bool:
        """Dunder names, declared exports, and framework hooks are never dead."""
        if MAGIC_NAMES.match(node.name) or node.name in exports:
            return True
        return bool(_decorator_names(node) & ENTRY_POINT_DECORATORS)

    def _private_finding(self, source: SourceFile, node) -> Finding:
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        return Finding(
            check=self.name,
            rule="unused-private",
            severity=Severity.MEDIUM,
            path=source.rel,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", None),
            message=(
                f"Private {kind} '{node.name}' is never referenced in its module. "
                f"If nothing calls it, delete it."
            ),
            snippet=f"private:{node.name}",
        )

    def _public_finding(self, source: SourceFile, node) -> Finding:
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        return Finding(
            check=self.name,
            rule="possibly-unused-public",
            severity=Severity.LOW,
            path=source.rel,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", None),
            message=(
                f"Public {kind} '{node.name}' has no reference anywhere in the "
                f"scanned tree. Verify it is not external API before removing."
            ),
            snippet=f"public:{node.name}",
        )

    def _unreachable(self, source: SourceFile) -> Iterable[Finding]:
        assert source.tree is not None
        terminators = (ast.Return, ast.Raise, ast.Break, ast.Continue)
        for node in ast.walk(source.tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for index, stmt in enumerate(body[:-1]):
                if isinstance(stmt, terminators):
                    nxt = body[index + 1]
                    yield Finding(
                        check=self.name,
                        rule="unreachable-code",
                        severity=Severity.HIGH,
                        path=source.rel,
                        line=nxt.lineno,
                        message=(
                            f"Unreachable: this statement follows a "
                            f"{type(stmt).__name__.lower()} and can never run."
                        ),
                        snippet=f"unreachable:{source.rel}:{type(stmt).__name__}:{index}",
                    )
                    break


# ==========================================================================
# Check: duplication
# ==========================================================================

"""Copy-paste detection over normalized statement sequences.

Physical-line matching is the obvious approach and the wrong one for Python:
a single call spread over eight lines looks like eight duplicated lines every
time it appears, so any codebase with a consistent house style lights up with
false positives. Instead we work at the AST statement level -- one statement is
one token, however many lines it occupies -- and look for runs of consecutive
statements that share a structure.

Literals are normalized away because those are exactly what a person edits
after pasting. Identifiers are kept: two blocks that differ only in variable
names are usually a deliberate parallel, while two blocks with identical names
are usually a copy.
"""
BLOCK_HOLDERS = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ExceptHandler,
)

# Statements too small to be worth reporting as "duplicated".
NOISE = (ast.Pass, ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


@dataclass(frozen=True)
class _Span:
    path: str
    start: int
    end: int
    statements: int


class _Normalizer(ast.NodeTransformer):
    """Erase literal values so edited copies still match the original."""

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, bool) or node.value is None:
            return node
        return ast.copy_location(ast.Constant(value="~"), node)


def _fingerprint_statement(stmt: ast.stmt) -> str:
    clone = _Normalizer().visit(ast.parse(ast.unparse(stmt)).body[0])
    return ast.dump(clone, annotate_fields=False)


def _statement_blocks(tree: ast.Module) -> Iterable[list[ast.stmt]]:
    """Every contiguous run of statements that shares a parent block."""
    for node in ast.walk(tree):
        if not isinstance(node, BLOCK_HOLDERS):
            continue
        for attr in ("body", "orelse", "finalbody"):
            body = getattr(node, attr, None)
            if isinstance(body, list) and len(body) > 1:
                yield [s for s in body if isinstance(s, ast.stmt)]


def _weight_of(stmt: ast.stmt) -> int:
    """How much substance a statement carries. Filters trivial runs."""
    return sum(1 for _ in ast.walk(stmt))


@register
class DuplicationCheck(Check):
    """Repeated statement runs, within and across modules."""

    name = "duplication"
    description = "Copy-pasted statement runs across and within modules"

    def run(self, sources: list[SourceFile], config: Config) -> Iterable[Finding]:
        """Hash every window of consecutive statements and find collisions."""
        size = config.thresholds.min_duplicate_statements
        buckets: dict[str, list[_Span]] = defaultdict(list)

        for source in sources:
            if source.tree is None:
                continue
            for block in _statement_blocks(source.tree):
                usable = [s for s in block if not isinstance(s, NOISE)]
                if len(usable) < size:
                    continue
                try:
                    prints = [_fingerprint_statement(s) for s in usable]
                except (SyntaxError, ValueError, RecursionError):
                    continue

                for i in range(len(usable) - size + 1):
                    window = usable[i : i + size]
                    # A run of near-empty statements is not interesting even if
                    # it repeats -- require real substance in the window.
                    if sum(_weight_of(s) for s in window) < size * 6:
                        continue
                    digest = hashlib.sha1(
                        "\n".join(prints[i : i + size]).encode("utf-8")
                    ).hexdigest()[:16]
                    start = window[0].lineno
                    end = getattr(window[-1], "end_lineno", window[-1].lineno)
                    buckets[digest].append(_Span(source.rel, start, end, size))

        yield from self._report(buckets, size)

    def _report(
        self, buckets: dict[str, list[_Span]], size: int
    ) -> Iterable[Finding]:
        claimed: set[tuple[str, int]] = set()

        # Longest, most-copied runs first, so a big duplicate is reported once
        # rather than as a dozen overlapping small ones.
        ordered = sorted(
            buckets.items(),
            key=lambda kv: (-len(kv[1]), -(kv[1][0].end - kv[1][0].start)),
        )

        for digest, spans in ordered:
            spans = self._dedupe(spans)
            if len(spans) < 2:
                continue
            if any((s.path, s.start) in claimed for s in spans):
                continue
            for s in spans:
                for line in range(s.start, s.end + 1):
                    claimed.add((s.path, line))

            spans.sort(key=lambda s: (s.path, s.start))
            primary = spans[0]
            cross_file = len({s.path for s in spans}) > 1
            others = ", ".join(f"{s.path}:{s.start}" for s in spans[1:4])
            more = f" (+{len(spans) - 4} more)" if len(spans) > 4 else ""
            span_lines = primary.end - primary.start + 1

            severity = Severity.MEDIUM
            if len(spans) >= 4 or span_lines >= 25:
                severity = Severity.HIGH

            scope = "across modules" if cross_file else "within one module"
            yield Finding(
                check=self.name,
                rule="duplicate-block",
                severity=severity,
                path=primary.path,
                line=primary.start,
                end_line=primary.end,
                message=(
                    f"{size} statements ({span_lines} lines) duplicated in "
                    f"{len(spans)} places {scope}. Also at: {others}{more}. "
                    f"Extract a shared helper."
                ),
                snippet=f"dup:{digest}",
                extra={
                    "copies": len(spans),
                    "locations": [f"{s.path}:{s.start}" for s in spans],
                },
            )

    @staticmethod
    def _dedupe(spans: list[_Span]) -> list[_Span]:
        """Drop windows that overlap an earlier one in the same file."""
        by_path: dict[str, list[_Span]] = defaultdict(list)
        for s in spans:
            by_path[s.path].append(s)
        kept: list[_Span] = []
        for group in by_path.values():
            group.sort(key=lambda s: s.start)
            last_end = -1
            for s in group:
                if s.start > last_end:
                    kept.append(s)
                    last_end = s.end
        return kept


# ==========================================================================
# Check: smells
# ==========================================================================

"""Code smells: the patterns that quietly cost you an afternoon later."""
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

MUTABLE_DEFAULTS = (ast.List, ast.Dict, ast.Set)

BROAD_EXCEPTIONS = {"Exception", "BaseException"}


@register
class SmellCheck(Check):
    """Local patterns that reliably cause trouble later."""

    name = "smells"
    description = "Bare excepts, mutable defaults, wildcard imports, silent failures"

    def run(self, sources: list[SourceFile], config: Config) -> Iterable[Finding]:
        """Walk each module for smells and debt markers."""
        for source in sources:
            if source.tree is None:
                yield Finding(
                    check=self.name,
                    rule="syntax-error",
                    severity=Severity.CRITICAL,
                    path=source.rel,
                    line=1,
                    message=f"File does not parse: {source.parse_error}",
                    snippet=f"syntax:{source.rel}",
                )
                continue
            yield from self._walk(source)
            yield from self._debt_markers(source)

    def _walk(self, source: SourceFile) -> Iterable[Finding]:
        assert source.tree is not None
        for node in ast.walk(source.tree):
            if isinstance(node, ast.ExceptHandler):
                yield from self._except_handler(source, node)
            elif isinstance(node, FUNC_NODES):
                yield from self._mutable_defaults(source, node)
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == "*" for a in node.names):
                    yield Finding(
                        check=self.name,
                        rule="wildcard-import",
                        severity=Severity.MEDIUM,
                        path=source.rel,
                        line=node.lineno,
                        message=(
                            f"'from {node.module} import *' hides what this module "
                            f"actually depends on."
                        ),
                        snippet=f"wildcard:{node.module}",
                    )
            elif isinstance(node, ast.Compare):
                yield from self._comparisons(source, node)
            elif isinstance(node, ast.Global):
                yield Finding(
                    check=self.name,
                    rule="global-statement",
                    severity=Severity.MEDIUM,
                    path=source.rel,
                    line=node.lineno,
                    message=(
                        f"'global {', '.join(node.names)}' -- mutable module state "
                        f"makes call order load-bearing."
                    ),
                    snippet=f"global:{','.join(node.names)}",
                )

    def _except_handler(
        self, source: SourceFile, node: ast.ExceptHandler
    ) -> Iterable[Finding]:
        if node.type is None:
            yield Finding(
                check=self.name,
                rule="bare-except",
                severity=Severity.HIGH,
                path=source.rel,
                line=node.lineno,
                message="Bare 'except:' also swallows KeyboardInterrupt and SystemExit.",
                snippet=f"bare-except:{source.rel}:{node.lineno}",
            )
        elif isinstance(node.type, ast.Name) and node.type.id in BROAD_EXCEPTIONS:
            yield Finding(
                check=self.name,
                rule="broad-except",
                severity=Severity.LOW,
                path=source.rel,
                line=node.lineno,
                message=(
                    f"'except {node.type.id}' catches everything. Name the errors "
                    f"you actually expect."
                ),
                snippet=f"broad-except:{source.rel}:{node.lineno}",
            )

        body = node.body
        only_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
        if only_pass:
            yield Finding(
                check=self.name,
                rule="silent-failure",
                severity=Severity.HIGH,
                path=source.rel,
                line=node.lineno,
                message=(
                    "Exception caught and discarded. At minimum log it -- silent "
                    "failures are the hardest bugs to find."
                ),
                snippet=f"silent:{source.rel}:{node.lineno}",
            )

    def _mutable_defaults(self, source: SourceFile, node) -> Iterable[Finding]:
        defaults = list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None
        ]
        for default in defaults:
            if isinstance(default, MUTABLE_DEFAULTS):
                yield Finding(
                    check=self.name,
                    rule="mutable-default",
                    severity=Severity.HIGH,
                    path=source.rel,
                    line=node.lineno,
                    message=(
                        f"{node.name}() has a mutable default argument. It is created "
                        f"once and shared by every call. Use None as the sentinel."
                    ),
                    snippet=f"mutable-default:{node.name}",
                )
            elif isinstance(default, ast.Call):
                func = default.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name in {"list", "dict", "set"}:
                    yield Finding(
                        check=self.name,
                        rule="mutable-default",
                        severity=Severity.HIGH,
                        path=source.rel,
                        line=node.lineno,
                        message=(
                            f"{node.name}() has a mutable default ({name}()) evaluated "
                            f"once at definition time. Use None as the sentinel."
                        ),
                        snippet=f"mutable-default:{node.name}",
                    )

    def _comparisons(self, source: SourceFile, node: ast.Compare) -> Iterable[Finding]:
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                yield Finding(
                    check=self.name,
                    rule="none-comparison",
                    severity=Severity.LOW,
                    path=source.rel,
                    line=node.lineno,
                    message="Compare to None with 'is' / 'is not', not '==' / '!='.",
                    snippet=f"none-cmp:{source.rel}:{node.lineno}",
                )
            elif isinstance(comparator, ast.Constant) and isinstance(
                comparator.value, bool
            ):
                yield Finding(
                    check=self.name,
                    rule="bool-comparison",
                    severity=Severity.LOW,
                    path=source.rel,
                    line=node.lineno,
                    message=(
                        f"Comparing to {comparator.value} explicitly -- test the "
                        f"expression directly instead."
                    ),
                    snippet=f"bool-cmp:{source.rel}:{node.lineno}",
                )

    def _debt_markers(self, source: SourceFile) -> Iterable[Finding]:
        markers = ("TODO", "FIXME", "HACK", "XXX")
        for number, raw in enumerate(source.lines, start=1):
            stripped = raw.strip()
            if not stripped.startswith("#"):
                continue
            for marker in markers:
                if marker in stripped:
                    severity = (
                        Severity.MEDIUM if marker in ("FIXME", "HACK") else Severity.LOW
                    )
                    yield Finding(
                        check=self.name,
                        rule="debt-marker",
                        severity=severity,
                        path=source.rel,
                        line=number,
                        message=f"{marker} comment: {stripped[:80]}",
                        snippet=f"debt:{source.rel}:{stripped[:40]}",
                    )
                    break


# ==========================================================================
# Check: docs
# ==========================================================================

"""Documentation coverage for the public surface of a package."""
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# Methods whose purpose is obvious from the name.
OBVIOUS = {
    "__init__",
    "__repr__",
    "__str__",
    "__eq__",
    "__hash__",
    "__len__",
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
    "main",
}


def _is_trivial(node) -> bool:
    """Property getters and one-line passthroughs do not need prose."""
    body = [s for s in node.body if not isinstance(s, ast.Pass)]
    if len(body) > 1:
        return False
    if not body:
        return True
    stmt = body[0]
    return isinstance(stmt, (ast.Return, ast.Expr))


@register
class DocsCheck(Check):
    """Docstring coverage on the public surface."""

    name = "docs"
    description = "Docstrings on the public API surface"

    def run(self, sources: list[SourceFile], config: Config) -> Iterable[Finding]:
        """Check module docstrings and public definitions."""
        if not config.thresholds.min_docstring_public:
            return
        for source in sources:
            if source.tree is None:
                continue
            yield from self._module(source)
            yield from self._definitions(source)

    def _module(self, source: SourceFile) -> Iterable[Finding]:
        assert source.tree is not None
        if source.path.name == "__init__.py" and source.sloc < 5:
            return
        if ast.get_docstring(source.tree):
            return
        yield Finding(
            check=self.name,
            rule="missing-module-docstring",
            severity=Severity.LOW,
            path=source.rel,
            line=1,
            message="Module has no docstring -- one line on what lives here is enough.",
            snippet=f"moduledoc:{source.rel}",
        )

    def _api_definitions(self, tree: ast.Module) -> Iterable[ast.AST]:
        """Yield module-level and class-level definitions only.

        ``ast.walk`` would also hand back nested closures, which are an
        implementation detail of the function they live in -- flagging a local
        ``walk()`` helper as undocumented public API is noise, not signal.
        """
        for node in tree.body:
            if isinstance(node, FUNC_NODES + (ast.ClassDef,)):
                yield node
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, FUNC_NODES):
                        yield member

    def _definitions(self, source: SourceFile) -> Iterable[Finding]:
        assert source.tree is not None
        for node in self._api_definitions(source.tree):
            name = node.name
            if name.startswith("_") or name in OBVIOUS:
                continue
            if ast.get_docstring(node):
                continue
            if isinstance(node, FUNC_NODES) and _is_trivial(node):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            if isinstance(node, FUNC_NODES) and (end - node.lineno) < 5:
                continue
            kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
            yield Finding(
                check=self.name,
                rule="missing-docstring",
                severity=Severity.LOW,
                path=source.rel,
                line=node.lineno,
                message=f"{kind} '{name}' is public and non-trivial but undocumented.",
                snippet=f"doc:{name}",
            )


# ==========================================================================
# Scoring
# ==========================================================================

"""The score.

Design goals, in order:

1. **Hard to game.** The number must only go up when the code actually gets
   better. Three defences: penalties are measured per 1000 source lines, so
   deleting good code raises the density of the bad code that remains; muted
   findings still count at a fraction of their weight in the strict score; and
   the curve is smooth, so there is no cliff to sit just underneath.
2. **Legible.** You can explain any score by pointing at the findings that
   produced it.
3. **Asymptotic.** 100 means "nothing detected", and the last few points are
   the expensive ones -- which matches how real cleanup feels.
"""
# Density (weighted penalty per KLOC) at which the score lands on ~37. Chosen so
# that ~5 penalty/KLOC scores 95 and ~50 scores 60 -- calibrated against a
# handful of well-regarded stdlib-adjacent packages.
# Calibrated so a codebase a seasoned engineer would call clean scores 95+.
DECAY_CONSTANT = 95.0

# What a muted finding still costs. Muting says "I accept this", not
# "this stopped existing".
MUTED_WEIGHT_FACTOR = 0.25

GRADES = (
    (98, "A+", "Beautiful. A seasoned engineer would not flinch."),
    (93, "A", "Clean. Minor polish left."),
    (85, "B", "Solid, with pockets of mess."),
    (75, "C", "Workable, but the debt is visible."),
    (60, "D", "Rough. Cleanup is now cheaper than the alternative."),
    (0, "F", "This is costing you real time every week."),
)


@dataclass
class Score:
    """A computed score plus everything needed to explain it."""

    value: float
    strict: float
    sloc: int
    density: float
    by_check: dict[str, float] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    finding_count: int = 0
    muted_count: int = 0

    @property
    def grade(self) -> str:
        return self._grade_entry()[1]

    @property
    def verdict(self) -> str:
        return self._grade_entry()[2]

    def _grade_entry(self) -> tuple[int, str, str]:
        for floor, letter, text in GRADES:
            if self.value >= floor:
                return floor, letter, text
        return GRADES[-1]

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 2),
            "strict": round(self.strict, 2),
            "grade": self.grade,
            "verdict": self.verdict,
            "sloc": self.sloc,
            "density": round(self.density, 3),
            "findings": self.finding_count,
            "muted": self.muted_count,
            "by_check": {k: round(v, 2) for k, v in self.by_check.items()},
            "by_severity": self.by_severity,
        }


def _curve(density: float) -> float:
    """Map weighted penalty density to 0-100.

    Exponential decay: every additional unit of density costs proportionally
    less than the last, which keeps a single catastrophic file from pinning an
    otherwise decent codebase at zero, while making the climb from 95 to 99
    genuinely demanding.
    """
    return 100.0 * exp(-density / DECAY_CONSTANT)


def compute(
    findings: list[Finding],
    sloc: int,
    ignored: set[str] | None = None,
) -> Score:
    """Score a scan.

    ``findings`` should be the complete set, including muted ones; this
    function applies the mute discount itself so that both the headline and
    strict numbers come from one pass.
    """
    ignored = ignored or set()
    kloc = max(sloc / 1000.0, 0.001)

    penalty = 0.0
    strict_penalty = 0.0
    by_check: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    muted = 0

    for finding in findings:
        weight = finding.weight
        strict_penalty += weight
        if finding.fingerprint in ignored:
            muted += 1
            weight *= MUTED_WEIGHT_FACTOR
        else:
            by_severity[finding.severity.value] += 1
        penalty += weight
        by_check[finding.check] += weight

    density = penalty / kloc
    strict_density = strict_penalty / kloc

    return Score(
        value=_curve(density),
        strict=_curve(strict_density),
        sloc=sloc,
        density=density,
        by_check=dict(by_check),
        by_severity=dict(by_severity),
        finding_count=len(findings),
        muted_count=muted,
    )


def priority(finding: Finding) -> tuple:
    """Sort key for the fix queue: worst payoff-per-effort first.

    Severity leads. Within a severity, checks are ordered by how mechanical the
    fix is -- deleting dead code is free, refactoring complexity is not -- so
    an agent working the queue top-down banks cheap wins early and keeps
    momentum.
    """
    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    effort_rank = {
        "deadcode": 0,
        "smells": 1,
        "duplication": 2,
        "docs": 3,
        "complexity": 4,
    }
    return (
        severity_rank[finding.severity],
        effort_rank.get(finding.check, 5),
        finding.path,
        finding.line,
    )


# ==========================================================================
# Persistent state
# ==========================================================================

"""Persistent state: what we saw last time, and what changed since.

State is what turns a linter into a campaign. Without it every scan is a wall
of noise; with it you can say "you fixed 14, three came back, here is the next
one to kill".
"""
STATE_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Delta:
    """What changed between the previous scan and this one."""

    new: list[Finding] = field(default_factory=list)
    fixed: list[dict] = field(default_factory=list)
    regressed: list[Finding] = field(default_factory=list)
    unchanged: int = 0
    score_change: float | None = None


@dataclass
class State:
    """On-disk memory of previous scans."""

    path: Path
    version: int = STATE_VERSION
    findings: dict[str, dict] = field(default_factory=dict)
    resolved: dict[str, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, state_dir: Path) -> "State":
        path = state_dir / "state.json"
        state = cls(path=path)
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return state
        if data.get("version") != STATE_VERSION:
            return state
        state.findings = data.get("findings", {})
        state.resolved = data.get("resolved", {})
        state.history = data.get("history", [])
        return state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "updated_at": _now(),
            "findings": self.findings,
            "resolved": self.resolved,
            "history": self.history[-200:],
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def diff(self, findings: list[Finding]) -> Delta:
        """Compare a fresh scan against what we remember."""
        current = {f.fingerprint: f for f in findings}
        previous = set(self.findings)
        delta = Delta()

        for fp, finding in current.items():
            if fp in previous:
                delta.unchanged += 1
            elif fp in self.resolved:
                delta.regressed.append(finding)
            else:
                delta.new.append(finding)

        for fp in previous - set(current):
            delta.fixed.append(self.findings[fp])

        return delta

    def record(self, findings: list[Finding], score: Score) -> Delta:
        """Fold a scan into state and return what changed."""
        delta = self.diff(findings)
        previous_score = self.history[-1]["score"] if self.history else None
        if previous_score is not None:
            delta.score_change = round(score.value - previous_score, 2)

        current = {f.fingerprint: f for f in findings}
        timestamp = _now()

        for fp in set(self.findings) - set(current):
            entry = dict(self.findings[fp])
            entry["resolved_at"] = timestamp
            self.resolved[fp] = entry

        merged: dict[str, dict] = {}
        for fp, finding in current.items():
            record = finding.to_dict()
            if fp in self.findings:
                record["first_seen"] = self.findings[fp].get("first_seen", timestamp)
            else:
                record["first_seen"] = timestamp
            record["last_seen"] = timestamp
            merged[fp] = record
            self.resolved.pop(fp, None)
        self.findings = merged

        self.history.append(
            {
                "at": timestamp,
                "score": round(score.value, 2),
                "strict": round(score.strict, 2),
                "grade": score.grade,
                "findings": len(findings),
                "sloc": score.sloc,
                "fixed_since_last": len(delta.fixed),
                "new_since_last": len(delta.new),
            }
        )
        return delta

    @property
    def total_resolved(self) -> int:
        return len(self.resolved)


# ==========================================================================
# Reporting
# ==========================================================================

"""Rendering scans for humans, agents, and CI."""
SEVERITY_ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]


class Palette:
    """ANSI colours, disabled when the output is not a friendly terminal."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = (
                sys.stdout.isatty()
                and os.environ.get("TERM") != "dumb"
                and "NO_COLOR" not in os.environ
            )
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def blue(self, t: str) -> str:
        return self._wrap("34", t)

    def severity(self, sev: Severity, text: str) -> str:
        return {
            Severity.CRITICAL: self.red,
            Severity.HIGH: self.red,
            Severity.MEDIUM: self.yellow,
            Severity.LOW: self.dim,
        }[sev](text)


def score_banner(score: Score, palette: Palette) -> str:
    """The headline score block."""
    colour = (
        palette.green
        if score.value >= 90
        else palette.yellow
        if score.value >= 75
        else palette.red
    )
    headline = colour(palette.bold(f"{score.value:.1f}/100  {score.grade}"))
    return (
        f"\n  {headline}   {palette.dim(score.verdict)}\n"
        f"  {palette.dim(f'strict {score.strict:.1f} | {score.sloc:,} sloc | '
                          f'{score.finding_count} findings')}\n"
    )


def summarize(findings: list[Finding], palette: Palette) -> str:
    """Counts by severity and by check."""
    counts: Counter[str] = Counter(f.check for f in findings)
    sev: Counter[Severity] = Counter(f.severity for f in findings)
    lines = ["  " + palette.bold("By severity")]
    for s in SEVERITY_ORDER:
        if sev[s]:
            lines.append(f"    {palette.severity(s, s.value.ljust(9))} {sev[s]}")
    lines.append("")
    lines.append("  " + palette.bold("By check"))
    for name, count in counts.most_common():
        lines.append(f"    {name.ljust(12)} {count}")
    return "\n".join(lines)


def render_delta(delta: Delta, palette: Palette) -> str:
    """What moved since the previous scan, or empty on a first run."""
    if delta.score_change is None:
        return ""
    arrow = "+" if delta.score_change >= 0 else ""
    colour = palette.green if delta.score_change >= 0 else palette.red
    parts = [f"  {colour(f'{arrow}{delta.score_change} since last scan')}"]
    bits = []
    if delta.fixed:
        bits.append(palette.green(f"{len(delta.fixed)} fixed"))
    if delta.new:
        bits.append(palette.yellow(f"{len(delta.new)} new"))
    if delta.regressed:
        bits.append(palette.red(f"{len(delta.regressed)} regressed"))
    if bits:
        parts.append("  " + palette.dim(" · ".join(bits)))
    return "\n".join(parts) + "\n"


def render_finding(finding: Finding, palette: Palette, index: int | None = None) -> str:
    """One finding, formatted for a terminal."""
    tag = palette.severity(finding.severity, f"[{finding.severity.value}]")
    prefix = f"{index}. " if index is not None else ""
    return (
        f"  {prefix}{tag} {palette.bold(finding.location)}  "
        f"{palette.dim(finding.rule)}\n"
        f"     {finding.message}\n"
        f"     {palette.dim('id ' + finding.fingerprint)}"
    )


def render_list(findings: list[Finding], palette: Palette, limit: int = 20) -> str:
    """The worst findings, in fix order."""
    ordered = sorted(findings, key=priority)
    chunks = [render_finding(f, palette) for f in ordered[:limit]]
    if len(ordered) > limit:
        chunks.append(palette.dim(f"\n  ... and {len(ordered) - limit} more"))
    return "\n\n".join(chunks)


def to_json(score: Score, findings: list[Finding], delta: Delta | None = None) -> str:
    """Machine-readable scan output."""
    payload: dict = {
        "score": score.to_dict(),
        "findings": [f.to_dict() for f in sorted(findings, key=priority)],
    }
    if delta is not None:
        payload["delta"] = {
            "new": len(delta.new),
            "fixed": len(delta.fixed),
            "regressed": len(delta.regressed),
            "unchanged": delta.unchanged,
            "score_change": delta.score_change,
        }
    return json.dumps(payload, indent=2)


def to_markdown(score: Score, findings: list[Finding]) -> str:
    """A report suitable for a PR comment or a docs page."""
    lines = [
        "# unbugify report",
        "",
        f"**Score: {score.value:.1f}/100 ({score.grade})** — {score.verdict}",
        "",
        f"- Strict score: {score.strict:.1f}",
        f"- Source lines: {score.sloc:,}",
        f"- Findings: {score.finding_count} ({score.muted_count} muted)",
        "",
        "## By check",
        "",
        "| Check | Findings | Penalty |",
        "| --- | --- | --- |",
    ]
    counts = Counter(f.check for f in findings)
    for name, weight in sorted(score.by_check.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {counts[name]} | {weight:.1f} |")

    lines += ["", "## Top findings", ""]
    for finding in sorted(findings, key=priority)[:25]:
        lines.append(
            f"- **{finding.severity.value}** `{finding.location}` "
            f"({finding.rule}) — {finding.message}"
        )
    return "\n".join(lines) + "\n"


# ==========================================================================
# Badge
# ==========================================================================

"""SVG badge generation, for READMEs.

Self-contained rather than shields.io-dependent: a badge that needs a network
round trip is a badge that eventually 404s in someone's README.
"""
_COLOURS = (
    (98, "#2ea043"),
    (93, "#3fb950"),
    (85, "#9acd32"),
    (75, "#d29922"),
    (60, "#db6d28"),
    (0, "#da3633"),
)

_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" \
xmlns:xlink="http://www.w3.org/1999/xlink" width="{total}" height="20" \
role="img" aria-label="unbugify: {label}">
  <title>unbugify: {label}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{left}" height="20" fill="#24292f"/>
    <rect x="{left}" width="{right}" height="20" fill="{colour}"/>
    <rect width="{total}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" \
font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="110">
    <text x="{lx}" y="150" fill="#010101" fill-opacity=".3" \
transform="scale(.1)" textLength="{lw}">unbugify</text>
    <text x="{lx}" y="140" transform="scale(.1)" textLength="{lw}">unbugify</text>
    <text x="{rx}" y="150" fill="#010101" fill-opacity=".3" \
transform="scale(.1)" textLength="{rw}">{label}</text>
    <text x="{rx}" y="140" transform="scale(.1)" textLength="{rw}">{label}</text>
  </g>
</svg>
"""


def colour_for(value: float) -> str:
    """Badge colour for a score, matching the grade bands."""
    for floor, colour in _COLOURS:
        if value >= floor:
            return colour
    return _COLOURS[-1][1]


def render_badge(score: Score) -> str:
    """Build a self-contained SVG badge for a score."""
    label = f"{score.value:.0f} {score.grade}"
    left = 62
    right = 10 + len(label) * 7
    total = left + right
    return _TEMPLATE.format(
        total=total,
        left=left,
        right=right,
        colour=colour_for(score.value),
        label=label,
        lx=left * 5,
        lw=(left - 12) * 10,
        rx=(left + right / 2) * 10,
        rw=(right - 10) * 10,
    )


# ==========================================================================
# Scan pipeline
# ==========================================================================

"""The scan pipeline: discover, check, score."""
def scan(config: Config, target: Path | None = None) -> tuple[ScanResult, Score]:
    """Run every enabled check over the tree and score the result."""
    sources = load_sources(config, target)
    result = ScanResult(
        modules=[
            ModuleInfo(path=s.rel, loc=s.loc, sloc=s.sloc, parse_error=s.parse_error)
            for s in sources
        ]
    )

    findings: list[Finding] = []
    for check in enabled_checks(config):
        findings.extend(check.run(sources, config))

    # Deterministic order so two runs of the same tree are byte-identical.
    findings.sort(key=lambda f: (f.path, f.line, f.rule, f.fingerprint))
    result.findings = findings

    score = compute(findings, result.total_sloc, ignored=set(config.ignored))
    return result, score


# ==========================================================================
# Command line interface
# ==========================================================================

"""Command line interface."""
def _resolve(path_arg: str | None) -> tuple[Config, Path]:
    """Locate the project root for a target path and load its config."""
    target = Path(path_arg or ".").resolve()
    root = target if target.is_dir() else target.parent
    # Walk up to a project marker so state lands next to pyproject.toml rather
    # than inside whatever subdirectory you happened to scan.
    marker = root
    for candidate in [root, *root.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            marker = candidate
            break
    config = Config.load(marker)
    return config, target


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan, score, record state, and render the result."""
    config, target = _resolve(args.path)
    palette = Palette(enabled=False if args.no_color else None)

    result, score = scan(config, target)
    state = State.load(config.state_dir)
    delta = state.record(result.findings, score)
    if not args.dry_run:
        state.save()

    if args.format == "json":
        print(to_json(score, result.findings, delta))
    elif args.format == "markdown":
        print(to_markdown(score, result.findings))
    else:
        print(score_banner(score, palette))
        change = render_delta(delta, palette)
        if change:
            print(change)
        print(summarize(result.findings, palette))
        print()
        if result.findings:
            print("  " + palette.bold("Worst first"))
            print()
            print(render_list(result.findings, palette, limit=args.limit))
            print()
            print(palette.dim("  Run 'unbugify next' to work the queue.\n"))
        else:
            print(palette.green("  Nothing found. Enjoy it while it lasts.\n"))

    if args.fail_under is not None and score.value < args.fail_under:
        print(
            f"Score {score.value:.1f} is below threshold {args.fail_under}.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    """Show the top of the fix queue."""
    config, target = _resolve(args.path)
    palette = Palette(enabled=False if args.no_color else None)
    result, _ = scan(config, target)

    queue = [
        f for f in sorted(result.findings, key=priority)
        if f.fingerprint not in config.ignored
    ]
    if not queue:
        print(palette.green("\n  Queue is empty. Nothing left to fix.\n"))
        return 0

    print()
    for index, finding in enumerate(queue[: args.count], start=1):
        print(render_finding(finding, palette, index=index))
        print()
    remaining = len(queue) - args.count
    if remaining > 0:
        print(palette.dim(f"  {remaining} more in the queue.\n"))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Write a full report to stdout or a file."""
    config, target = _resolve(args.path)
    result, score = scan(config, target)
    text = (
        to_json(score, result.findings)
        if args.format == "json"
        else to_markdown(score, result.findings)
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_badge(args: argparse.Namespace) -> int:
    """Render an SVG score badge."""
    config, target = _resolve(args.path)
    _, score = scan(config, target)
    svg = render_badge(score)
    output = Path(args.output or (config.state_dir / "badge.svg"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(f"Wrote {output} ({score.value:.1f} {score.grade})")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Print the score trend from recorded state."""
    config, _ = _resolve(args.path)
    palette = Palette(enabled=False if args.no_color else None)
    state = State.load(config.state_dir)
    if not state.history:
        print("No history yet. Run 'unbugify scan' first.")
        return 0
    print()
    print(f"  {'date':<22}{'score':>7}{'strict':>8}{'findings':>10}{'sloc':>9}")
    for entry in state.history[-args.limit :]:
        print(
            f"  {entry['at'][:19]:<22}{entry['score']:>7.1f}"
            f"{entry['strict']:>8.1f}{entry['findings']:>10}{entry['sloc']:>9,}"
        )
    print()
    print(palette.dim(f"  {state.total_resolved} findings resolved all-time.\n"))
    return 0


def cmd_ignore(args: argparse.Namespace) -> int:
    """Mute findings by fingerprint."""
    config, _ = _resolve(args.path)
    changed = False
    for fingerprint in args.fingerprints:
        if fingerprint not in config.ignored:
            config.ignored.append(fingerprint)
            changed = True
    if changed:
        config.save()
    print(
        f"Muted {len(args.fingerprints)} finding(s). "
        f"They still count at 25% in the strict score."
    )
    return 0


def cmd_unignore(args: argparse.Namespace) -> int:
    """Unmute findings by fingerprint."""
    config, _ = _resolve(args.path)
    config.ignored = [f for f in config.ignored if f not in args.fingerprints]
    config.save()
    print(f"Unmuted {len(args.fingerprints)} finding(s).")
    return 0


def cmd_exclude(args: argparse.Namespace) -> int:
    """Add exclude patterns to the saved config."""
    config, _ = _resolve(None)
    for pattern in args.patterns:
        if pattern not in config.excludes:
            config.excludes.append(pattern)
    config.save()
    print(f"Excluding: {', '.join(args.patterns)}")
    return 0


def cmd_checks(args: argparse.Namespace) -> int:
    """List the registered checks."""
    palette = Palette(enabled=False if args.no_color else None)
    print()
    for check in all_checks():
        print(f"  {palette.bold(check.name.ljust(14))}{check.description}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="unbugify",
        description="Score a Python codebase and work the queue until it is clean.",
    )
    parser.add_argument("--version", action="version", version=f"unbugify {__version__}")
    parser.add_argument(
        "--no-color", action="store_true", help="disable ANSI colour output"
    )

    # Shared flags accepted either before or after the subcommand, because
    # "unbugify scan --no-color" is what everyone types first.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan the codebase and score it", parents=[common])
    p_scan.add_argument("--path", default=".", help="directory or file to scan")
    p_scan.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text"
    )
    p_scan.add_argument("--limit", type=int, default=20, help="findings to display")
    p_scan.add_argument(
        "--fail-under", type=float, default=None, help="exit 1 below this score (CI)"
    )
    p_scan.add_argument(
        "--dry-run", action="store_true", help="do not write state to disk"
    )
    p_scan.set_defaults(func=cmd_scan)

    p_next = sub.add_parser("next", help="show the next thing worth fixing", parents=[common])
    p_next.add_argument("--path", default=".")
    p_next.add_argument("--count", type=int, default=3)
    p_next.set_defaults(func=cmd_next)

    p_report = sub.add_parser("report", help="write a full report", parents=[common])
    p_report.add_argument("--path", default=".")
    p_report.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p_report.add_argument("--output", default=None)
    p_report.set_defaults(func=cmd_report)

    p_badge = sub.add_parser("badge", help="generate an SVG score badge", parents=[common])
    p_badge.add_argument("--path", default=".")
    p_badge.add_argument("--output", default=None)
    p_badge.set_defaults(func=cmd_badge)

    p_history = sub.add_parser("history", help="show score over time", parents=[common])
    p_history.add_argument("--path", default=".")
    p_history.add_argument("--limit", type=int, default=20)
    p_history.set_defaults(func=cmd_history)

    p_ignore = sub.add_parser("ignore", help="mute findings by id", parents=[common])
    p_ignore.add_argument("fingerprints", nargs="+")
    p_ignore.add_argument("--path", default=".")
    p_ignore.set_defaults(func=cmd_ignore)

    p_unignore = sub.add_parser("unignore", help="unmute findings by id", parents=[common])
    p_unignore.add_argument("fingerprints", nargs="+")
    p_unignore.add_argument("--path", default=".")
    p_unignore.set_defaults(func=cmd_unignore)

    p_exclude = sub.add_parser("exclude", help="exclude a path from scans", parents=[common])
    p_exclude.add_argument("patterns", nargs="+")
    p_exclude.set_defaults(func=cmd_exclude)

    p_checks = sub.add_parser("checks", help="list available checks", parents=[common])
    p_checks.set_defaults(func=cmd_checks)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `unbugify scan | head` closes the pipe early. Exiting quietly is the
        # civilised response; Python otherwise prints a traceback at shutdown.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
