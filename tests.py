"""Integration tests for CodeAtlas. Run with: pytest tests.py -v"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SCRIPT = ROOT / "codeatlas.py"


@pytest.fixture
def sample_repo(tmp_path):
    """Build a tiny fixture repo with a class, a caller, and a cycle."""
    (tmp_path / "app.py").write_text(
        '"""Tiny fixture for tests."""\n'
        "\n"
        "\n"
        "class Greeter:\n"
        '    """Greets people."""\n'
        "\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n"
        "\n"
        "    def hello(self) -> str:\n"
        "        return f\"hello {self.name}\"\n"
        "\n"
        "\n"
        "def build_greeter() -> Greeter:\n"
        "    return Greeter(\"world\")\n"
        "\n"
        "\n"
        "def run() -> str:\n"
        "    g = build_greeter()\n"
        "    return g.hello()\n"
    )
    return tmp_path


def run_cli(*args, cwd=None):
    """Run codeatlas.py as a subprocess; return (rc, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(cwd or ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def test_index_builds_graph(sample_repo):
    rc, out, _ = run_cli("index", str(sample_repo))
    assert rc == 0
    assert "Saved:" in out
    graph_file = sample_repo / ".codeatlas" / "graph.json"
    assert graph_file.exists()

    data = json.loads(graph_file.read_text())
    assert any(s["name"] == "Greeter" for s in data["symbols"])
    assert any(s["name"] == "build_greeter" for s in data["symbols"])
    assert any(e["kind"] == "calls" for e in data["edges"])


def test_stats_json_output(sample_repo):
    rc, out, _ = run_cli("stats", str(sample_repo), "--json")
    assert rc == 0
    payload = json.loads(out)
    assert payload["symbols"] > 0
    assert payload["files"] == 1
    assert payload["symbols_by_kind"]["class"] == 1


def test_callers_resolution(sample_repo):
    """Test that `g.hello()` in `run` resolves to `Greeter.hello`."""
    run_cli("index", str(sample_repo))
    rc, out, _ = run_cli("callers", "Greeter.hello",
                         "--path", str(sample_repo), "--json")
    assert rc == 0
    payload = json.loads(out)
    caller_names = [s["symbol"] for s in payload["symbols"]]
    # run() calls g.hello() where g = build_greeter() -> Greeter
    assert any("run" in n for n in caller_names)


def test_impact_analysis(sample_repo):
    run_cli("index", str(sample_repo))
    rc, out, _ = run_cli("impact", "Greeter.hello",
                         "--path", str(sample_repo), "--json")
    assert rc == 0
    payload = json.loads(out)
    assert payload["transitive"] >= 1


def test_dead_code_detection(sample_repo):
    (sample_repo / "orphan.py").write_text(
        "def never_called():\n"
        "    return 42\n"
    )
    run_cli("index", str(sample_repo))
    rc, out, _ = run_cli("dead-code", "--path", str(sample_repo), "--json")
    assert rc == 0
    dead = json.loads(out)
    names = [d["symbol"] for d in dead]
    assert any("never_called" in n for n in names)


def test_no_false_cycle(sample_repo):
    rc, out, _ = run_cli("cycles", "--path", str(sample_repo), "--json")
    assert rc == 0
    cycles = json.loads(out)
    assert cycles == []


def test_graph_mermaid_output(sample_repo):
    run_cli("index", str(sample_repo))
    rc, out, _ = run_cli("graph", "--path", str(sample_repo))
    assert rc == 0
    assert out.startswith("flowchart LR")


def test_ask_dry_run(sample_repo):
    run_cli("index", str(sample_repo))
    rc, out, _ = run_cli("ask", "how does greeter work",
                         "--path", str(sample_repo), "--dry-run")
    assert rc == 0
    assert "dry run" in out.lower()


def test_cache_staleness(sample_repo):
    run_cli("index", str(sample_repo))
    # Touch a source file to make cache stale
    import time
    time.sleep(0.05)
    (sample_repo / "newfile.py").write_text("def added_fn():\n    pass\n")
    rc, out, _ = run_cli("list", "--path", str(sample_repo), "--json")
    assert rc == 0
    symbols = json.loads(out)
    assert any("added_fn" in s["symbol"] for s in symbols), \
        "Staleness detection failed — new file not picked up"


def test_version():
    rc, out, _ = run_cli("version")
    assert rc == 0
    assert "codeatlas" in out