"""Tests for unbugify.

Every check gets a positive case and, where it matters, a negative one -- a
detector that only ever fires is as useless as one that never does. The scoring
tests pin the properties that make the number worth trusting: you cannot raise
it by deleting code, and you cannot mute your way to an A.
"""

from __future__ import annotations

from pathlib import Path
import json
import pytest
import textwrap

from unbugify import (
    Config,
    Finding,
    Severity,
    State,
    Thresholds,
    colour_for,
    compute,
    main,
    priority,
    render_badge,
    scan,
)

def write(tmp_path: Path, name: str, code: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code).lstrip(), encoding="utf-8")
    return path


def rules(tmp_path: Path) -> list[str]:
    config = Config(root=tmp_path)
    result, _ = scan(config, tmp_path)
    return [f.rule for f in result.findings]


class TestComplexity:
    def test_flat_elif_chain_is_not_deep_nesting(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def dispatch(kind):
                \"\"\"Dispatch on kind.\"\"\"
                if kind == 1:
                    return "a"
                elif kind == 2:
                    return "b"
                elif kind == 3:
                    return "c"
                elif kind == 4:
                    return "d"
                elif kind == 5:
                    return "e"
                return None
            """,
        )
        assert "deep-nesting" not in rules(tmp_path)

    def test_genuinely_nested_code_is_flagged(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def tangled(rows):
                \"\"\"Deeply nested loop.\"\"\"
                for row in rows:
                    if row:
                        for cell in row:
                            if cell:
                                while cell:
                                    cell -= 1
                return rows
            """,
        )
        assert "deep-nesting" in rules(tmp_path)

    def test_too_many_parameters_ignores_self(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            class Thing:
                \"\"\"A thing.\"\"\"

                def method(self, a, b, c, d, e):
                    \"\"\"Five real params is at the limit, not over it.\"\"\"
                    return a + b + c + d + e
            """,
        )
        assert "too-many-parameters" not in rules(tmp_path)


class TestDeadCode:
    def test_unused_import(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"
            import os
            import sys

            print(sys.argv)
            """,
        )
        found = rules(tmp_path)
        assert "unused-import" in found

    def test_import_used_only_in_annotation_is_not_dead(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"
            from pathlib import Path

            def load(p: Path) -> str:
                \"\"\"Read a file.\"\"\"
                return p.read_text()
            """,
        )
        assert "unused-import" not in rules(tmp_path)

    def test_unreachable_code(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def f():
                \"\"\"Return early.\"\"\"
                return 1
                print("never")
            """,
        )
        assert "unreachable-code" in rules(tmp_path)

    def test_unused_private_function(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def _helper():
                \"\"\"Nobody calls this.\"\"\"
                return 1

            def public():
                \"\"\"Entry point.\"\"\"
                return 2
            """,
        )
        assert "unused-private" in rules(tmp_path)

    def test_private_used_elsewhere_in_module_is_alive(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def _helper():
                \"\"\"Called below.\"\"\"
                return 1

            def public():
                \"\"\"Entry point.\"\"\"
                return _helper()
            """,
        )
        assert "unused-private" not in rules(tmp_path)

    def test_symbol_used_in_another_module_is_alive(self, tmp_path):
        write(
            tmp_path,
            "a.py",
            """
            \"\"\"Provider.\"\"\"

            def shared():
                \"\"\"Used by b.\"\"\"
                return 1
            """,
        )
        write(
            tmp_path,
            "b.py",
            """
            \"\"\"Consumer.\"\"\"
            from a import shared

            print(shared())
            """,
        )
        assert "possibly-unused-public" not in rules(tmp_path)


class TestSmells:
    @pytest.mark.parametrize(
        "code,rule",
        [
            ("def f(x=[]):\n    return x\n", "mutable-default"),
            ("def f(x=dict()):\n    return x\n", "mutable-default"),
            ("try:\n    pass\nexcept:\n    pass\n", "bare-except"),
            ("try:\n    x = 1\nexcept ValueError:\n    pass\n", "silent-failure"),
            ("from os import *\n", "wildcard-import"),
            ("x = 1\nif x == None:\n    pass\n", "none-comparison"),
            ("# TODO: fix this\nx = 1\n", "debt-marker"),
        ],
    )
    def test_smell_detected(self, tmp_path, code, rule):
        write(tmp_path, "m.py", '"""Module."""\n' + code)
        assert rule in rules(tmp_path)

    def test_syntax_error_is_critical(self, tmp_path):
        write(tmp_path, "m.py", "def broken(:\n")
        assert "syntax-error" in rules(tmp_path)


class TestDuplication:
    def test_real_copy_paste_is_caught(self, tmp_path):
        block = """
                total = 0
                for item in items:
                    total += item.value
                    if total > limit:
                        break
                result = total * 2
                logger.info(result)
        """
        write(
            tmp_path,
            "a.py",
            f'''
            """A."""

            def first(items, limit, logger):
                """Do the thing."""
{block}
                return result
            ''',
        )
        write(
            tmp_path,
            "b.py",
            f'''
            """B."""

            def second(items, limit, logger):
                """Do the same thing."""
{block}
                return result
            ''',
        )
        assert "duplicate-block" in rules(tmp_path)

    def test_multiline_calls_are_not_duplicates(self, tmp_path):
        # The classic false positive: a house style where every call is spread
        # over many lines with identical keyword names.
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def build(make):
                \"\"\"Build several different things.\"\"\"
                a = make(
                    name="alpha",
                    kind="one",
                    size=1,
                )
                b = make(
                    name="beta",
                    kind="two",
                    size=2,
                )
                c = make(
                    name="gamma",
                    kind="three",
                    size=3,
                )
                return a, b, c
            """,
        )
        assert "duplicate-block" not in rules(tmp_path)


class TestDocs:
    def test_missing_docstring_on_public_function(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def important(a, b):
                total = a + b
                total *= 2
                total -= 1
                total += 3
                return total
            """,
        )
        assert "missing-docstring" in rules(tmp_path)

    def test_trivial_function_needs_no_docstring(self, tmp_path):
        write(
            tmp_path,
            "m.py",
            """
            \"\"\"Module.\"\"\"

            def name(self):
                return self._name
            """,
        )
        assert "missing-docstring" not in rules(tmp_path)

def make_finding(rule: str = "r", severity: Severity = Severity.MEDIUM, line: int = 1):
    return Finding(
        check="smells",
        rule=rule,
        severity=severity,
        path="m.py",
        line=line,
        message="msg",
        snippet=f"{rule}:{line}",
    )


class TestScoring:
    def test_clean_codebase_scores_100(self):
        assert compute([], 1000).value == pytest.approx(100.0)

    def test_score_never_negative(self):
        findings = [make_finding(severity=Severity.CRITICAL, line=i) for i in range(500)]
        score = compute(findings, 100)
        assert 0.0 <= score.value <= 100.0

    def test_more_findings_lower_score(self):
        few = compute([make_finding(line=i) for i in range(3)], 1000)
        many = compute([make_finding(line=i) for i in range(30)], 1000)
        assert many.value < few.value

    def test_deleting_good_code_does_not_raise_score(self):
        """The core anti-gaming property.

        Same findings, smaller codebase -> higher density -> worse score. You
        cannot dilute your problems away by deleting the code that works.
        """
        findings = [make_finding(line=i) for i in range(10)]
        big = compute(findings, 5000)
        small = compute(findings, 1000)
        assert small.value < big.value

    def test_muting_helps_headline_but_not_strict(self):
        findings = [make_finding(line=i) for i in range(10)]
        ignored = {findings[0].fingerprint, findings[1].fingerprint}
        muted = compute(findings, 1000, ignored=ignored)
        plain = compute(findings, 1000)
        assert muted.value > plain.value
        assert muted.strict == pytest.approx(plain.strict)
        assert muted.muted_count == 2

    def test_severity_weights_ordered(self):
        assert (
            Severity.CRITICAL.weight
            > Severity.HIGH.weight
            > Severity.MEDIUM.weight
            > Severity.LOW.weight
        )

    def test_grades_track_value(self):
        assert compute([], 1000).grade == "A+"
        rough = compute([make_finding(severity=Severity.CRITICAL, line=i)
                         for i in range(40)], 1000)
        assert rough.grade in {"D", "F"}

    def test_priority_puts_critical_first(self):
        low = make_finding(severity=Severity.LOW, line=1)
        crit = make_finding(severity=Severity.CRITICAL, line=2)
        assert sorted([low, crit], key=priority)[0] is crit


class TestFingerprint:
    def test_stable_across_line_moves(self):
        a = Finding("smells", "bare-except", Severity.HIGH, "m.py", 10, "x", "snip")
        b = Finding("smells", "bare-except", Severity.HIGH, "m.py", 99, "x", "snip")
        assert a.fingerprint == b.fingerprint

    def test_differs_across_files(self):
        a = Finding("smells", "bare-except", Severity.HIGH, "a.py", 1, "x", "snip")
        b = Finding("smells", "bare-except", Severity.HIGH, "b.py", 1, "x", "snip")
        assert a.fingerprint != b.fingerprint


class TestState:
    def test_roundtrip(self, tmp_path):
        state = State.load(tmp_path)
        findings = [make_finding(line=i) for i in range(3)]
        score = compute(findings, 1000)
        state.record(findings, score)
        state.save()

        reloaded = State.load(tmp_path)
        assert len(reloaded.findings) == 3
        assert len(reloaded.history) == 1

    def test_detects_fixed_and_new(self, tmp_path):
        state = State.load(tmp_path)
        first = [make_finding(line=i) for i in range(3)]
        state.record(first, compute(first, 1000))

        second = [make_finding(line=i) for i in (1, 2, 9)]
        delta = state.record(second, compute(second, 1000))
        assert len(delta.fixed) == 1
        assert len(delta.new) == 1
        assert delta.unchanged == 2

    def test_detects_regression(self, tmp_path):
        state = State.load(tmp_path)
        original = [make_finding(line=1)]
        state.record(original, compute(original, 1000))
        state.record([], compute([], 1000))
        delta = state.record(original, compute(original, 1000))
        assert len(delta.regressed) == 1
        assert not delta.new

    def test_score_change_recorded(self, tmp_path):
        state = State.load(tmp_path)
        many = [make_finding(line=i) for i in range(20)]
        state.record(many, compute(many, 1000))
        delta = state.record([], compute([], 1000))
        assert delta.score_change > 0


class TestConfig:
    def test_default_excludes_apply(self, tmp_path):
        config = Config(root=tmp_path)
        assert config.is_excluded(tmp_path / ".venv" / "lib" / "x.py")
        assert config.is_excluded(tmp_path / "node_modules" / "y.py")
        assert not config.is_excluded(tmp_path / "src" / "app.py")

    def test_pyproject_thresholds(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            textwrap.dedent(
                """
                [tool.unbugify]
                exclude = ["vendor"]
                include_tests = true

                [tool.unbugify.thresholds]
                max_cyclomatic = 25
                """
            ),
            encoding="utf-8",
        )
        config = Config.load(tmp_path)
        assert config.thresholds.max_cyclomatic == 25
        assert config.include_tests is True
        assert "vendor" in config.excludes

    def test_save_and_reload(self, tmp_path):
        config = Config(root=tmp_path)
        config.ignored.append("deadbeef1234")
        config.excludes.append("vendor")
        config.save()

        reloaded = Config.load(tmp_path)
        assert "deadbeef1234" in reloaded.ignored
        assert "vendor" in reloaded.excludes

    def test_thresholds_defaults_are_sane(self):
        t = Thresholds()
        assert t.max_cyclomatic > 0
        assert t.max_function_lines > t.max_parameters


class TestBadge:
    def test_renders_valid_svg(self):
        svg = render_badge(compute([], 1000))
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "unbugify" in svg

    def test_colour_bands(self):
        assert colour_for(99) != colour_for(50)
        assert colour_for(10) == colour_for(0)


class TestCLI:
    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / "m.py").write_text(
            '"""M."""\nimport os\n\n\ndef f():\n    """F."""\n    return 1\n',
            encoding="utf-8",
        )
        return tmp_path

    def test_scan_text(self, project, capsys, monkeypatch):
        monkeypatch.chdir(project)
        assert main(["scan", "--no-color"]) == 0
        assert "/100" in capsys.readouterr().out

    def test_scan_json_is_parseable(self, project, capsys, monkeypatch):
        monkeypatch.chdir(project)
        main(["scan", "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert "score" in payload and "findings" in payload

    def test_fail_under_exits_nonzero(self, project, capsys, monkeypatch):
        monkeypatch.chdir(project)
        assert main(["scan", "--fail-under", "101"]) == 1

    def test_fail_under_passes_when_met(self, project, monkeypatch):
        monkeypatch.chdir(project)
        assert main(["scan", "--fail-under", "0"]) == 0

    def test_next_runs(self, project, capsys, monkeypatch):
        monkeypatch.chdir(project)
        assert main(["next", "--no-color"]) == 0

    def test_badge_written(self, project, tmp_path, monkeypatch):
        monkeypatch.chdir(project)
        out = tmp_path / "b.svg"
        assert main(["badge", "--output", str(out)]) == 0
        assert out.read_text().startswith("<svg")

    def test_history_after_scan(self, project, capsys, monkeypatch):
        monkeypatch.chdir(project)
        main(["scan"])
        capsys.readouterr()
        main(["history"])
        assert "score" in capsys.readouterr().out

    def test_dry_run_writes_no_state(self, project, monkeypatch):
        monkeypatch.chdir(project)
        main(["scan", "--dry-run"])
        assert not (project / ".unbugify" / "state.json").exists()

    def test_scan_is_deterministic(self, project, capsys, monkeypatch):
        monkeypatch.chdir(project)
        main(["scan", "--format", "json", "--dry-run"])
        first = capsys.readouterr().out
        main(["scan", "--format", "json", "--dry-run"])
        second = capsys.readouterr().out
        assert json.loads(first)["findings"] == json.loads(second)["findings"]
