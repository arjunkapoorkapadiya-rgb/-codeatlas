#!/usr/bin/env python3
"""
CodeAtlas — AI-powered code intelligence CLI.

Parses a Python repo into a semantic graph, then lets you:
  * ask natural-language questions about it (LLM-backed, citation-grounded)
  * analyze impact of changes, call cycles, hotspots, dead code
  * inspect any symbol with syntax-highlighted source and call graph
  * export Mermaid dependency diagrams for READMEs

Usage:
    codeatlas index [path]                    Parse a repo, build the graph
    codeatlas stats [path]                    Quick summary
    codeatlas explain <symbol>                Full detail on one symbol
    codeatlas list [--kind <kind>]            List symbols
    codeatlas search <pattern>                Substring search over names
    codeatlas callers <symbol>                Who calls this?
    codeatlas callees <symbol>                What does this call?
    codeatlas tree <symbol> [--depth N]       Recursive callee tree
    codeatlas impact <symbol> [--depth N]     Recursive caller tree
    codeatlas hotspots [--top N]              Rank by caller count
    codeatlas dead-code                       Find uncalled functions
    codeatlas cycles                          Detect call cycles (SCCs)
    codeatlas graph [--symbol X] [--depth N]  Mermaid diagram
    codeatlas ask "<question>"                AI Q&A with citations

Global flags:
    --path <repo>     Point at any repo (default: current dir)
    --json            Machine-readable output
    --verbose / -v    Diagnostic logging to stderr

Ask-specific flags:
    --backend ollama|anthropic|openai   Explicit LLM backend
    --model <name>                      Override default model
    --dry-run                           Show retrieved context, skip LLM
    --why                               Show why each symbol was retrieved
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.text import Text
    from rich.markdown import Markdown
    from rich import box as rich_box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

__version__ = "1.0.0"

SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", "dist", "build", ".tox", ".idea", ".vscode",
    "site-packages", ".codeatlas",
}

GRAPH_DIR = ".codeatlas"
GRAPH_FILE = "graph.json"

# Never resolve calls that begin with these — stdlib calls should not
# accidentally match same-named user symbols (e.g. ast.parse vs Parser.parse).
_STDLIB_PREFIXES = (
    "ast.", "os.", "sys.", "json.", "re.", "io.", "time.", "datetime.",
    "pathlib.", "typing.", "dataclasses.", "collections.", "functools.",
    "itertools.", "math.", "random.", "string.", "subprocess.",
    "logging.", "enum.", "abc.", "contextlib.", "unittest.",
    "asyncio.", "threading.", "multiprocessing.", "socket.", "shutil.",
    "tempfile.", "textwrap.", "traceback.", "copy.", "pickle.",
    "hashlib.", "base64.", "urllib.", "http.", "email.",
    "warnings.", "inspect.", "importlib.", "platform.", "uuid.",
    "csv.", "sqlite3.", "argparse.", "configparser.", "getpass.",
    "builtins.",
)


# =============================================================================
# Data Model
# =============================================================================


class NodeKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeKind(str, Enum):
    CONTAINS = "contains"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"


@dataclass
class Symbol:
    id: str
    name: str
    qualified_name: str
    kind: NodeKind
    file: str
    lineno: int
    end_lineno: int
    col_offset: int = 0
    parent: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Symbol":
        d = dict(d)
        d["kind"] = NodeKind(d["kind"])
        return cls(**d)


@dataclass
class Edge:
    src: str
    dst: str
    kind: EdgeKind

    def to_dict(self) -> dict[str, str]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind.value}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Edge":
        return cls(d["src"], d["dst"], EdgeKind(d["kind"]))


@dataclass
class UnresolvedCall:
    src_id: str
    raw_name: str
    attr: str
    lineno: int


@dataclass
class CodeGraph:
    symbols: dict[str, Symbol] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    root: str = ""
    generated_at: float = field(default_factory=time.time)
    maybe_dynamic_calls: set[str] = field(default_factory=set)

    def add_symbol(self, sym: Symbol) -> None:
        self.symbols[sym.id] = sym

    def add_edge(self, src: str, dst: str, kind: EdgeKind) -> None:
        if src == dst:
            return
        if src not in self.symbols or dst not in self.symbols:
            return
        for e in self.edges:
            if e.src == src and e.dst == dst and e.kind == kind:
                return
        self.edges.append(Edge(src, dst, kind))

    def get(self, sym_id: str) -> Optional[Symbol]:
        return self.symbols.get(sym_id)

    def children_of(self, sym_id: str) -> list[Symbol]:
        return [s for e in self.edges
                if e.kind == EdgeKind.CONTAINS and e.src == sym_id
                for s in [self.symbols.get(e.dst)]
                if s is not None]

    def callers_of(self, sym_id: str) -> list[Symbol]:
        return [s for e in self.edges
                if e.kind == EdgeKind.CALLS and e.dst == sym_id
                for s in [self.symbols.get(e.src)]
                if s is not None]

    def callees_of(self, sym_id: str) -> list[Symbol]:
        return [s for e in self.edges
                if e.kind == EdgeKind.CALLS and e.src == sym_id
                for s in [self.symbols.get(e.dst)]
                if s is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "version": __version__,
            "generated_at": self.generated_at,
            "symbols": [s.to_dict() for s in self.symbols.values()],
            "edges": [e.to_dict() for e in self.edges],
            "maybe_dynamic_calls": sorted(self.maybe_dynamic_calls),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "CodeGraph":
        data = json.loads(path.read_text())
        g = cls(root=data["root"], generated_at=data.get("generated_at", 0))
        for d in data["symbols"]:
            sym = Symbol.from_dict(d)
            g.symbols[sym.id] = sym
        g.edges = [Edge.from_dict(d) for d in data["edges"]]
        g.maybe_dynamic_calls = set(data.get("maybe_dynamic_calls", []))
        return g


# =============================================================================
# Parser — AST → symbols → edges (3-phase: index, parse, resolve)
# =============================================================================


class Parser:
    def __init__(self, root: Path, verbose: bool = False):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"path not found: {self.root}")
        self.verbose = verbose
        self.graph = CodeGraph(root=str(self.root))

        self._files: list[Path] = []
        self._pending_assignments: list = []
        self._module_id_by_dotted: dict[str, str] = {}
        self._unresolved_calls: list[UnresolvedCall] = []
        self._imports_by_module: dict[str, list[dict]] = {}
        self._maybe_dynamic_names: set[str] = set()

        # Local type tracking (populated during parse; used during resolve)
        self._scope_vars: dict[str, dict[str, str]] = {}
        # func_id_or_module_id → {var_name → class_name}
        self._class_attrs: dict[str, dict[str, str]] = {}
        # class_id → {attr_name → class_name}

    # --- public entry ---

    def parse(self) -> CodeGraph:
        self._index_files()
        if self.verbose:
            print(f"[codeatlas] discovered {len(self._files)} files", file=sys.stderr)
        for f in self._files:
            try:
                self._parse_file(f)
            except SyntaxError as exc:
                if self.verbose:
                    print(f"[codeatlas] skip (syntax): {f}: {exc}", file=sys.stderr)
            except Exception as exc:
                if self.verbose:
                    print(f"[codeatlas] skip (error): {f}: {exc}", file=sys.stderr)
        self._resolve_pending_assignments()
        self._resolve_imports()
        self._resolve_calls()
        self.graph.maybe_dynamic_calls = set(self._maybe_dynamic_names)
        return self.graph

    # --- phase 1: discovery ---

    def _index_files(self) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = Path(dirpath) / fn
                rel = p.relative_to(self.root)
                rel_str = str(rel).replace(os.sep, "/")
                self._files.append(p)
                dotted = _path_to_dotted(rel_str)
                self._module_id_by_dotted[dotted] = rel_str

    # --- phase 2: per-file walk ---

    def _parse_file(self, path: Path) -> None:
        rel = str(path.relative_to(self.root)).replace(os.sep, "/")
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))

        module_id = rel
        module_symbol = Symbol(
            id=module_id,
            name=path.stem,
            qualified_name=_path_to_dotted(rel),
            kind=NodeKind.MODULE,
            file=rel,
            lineno=1,
            end_lineno=getattr(tree, "end_lineno", None) or source.count("\n") + 1,
            docstring=ast.get_docstring(tree),
        )
        self.graph.add_symbol(module_symbol)

        self._imports_by_module[module_id] = []
        self._walk_body(tree.body, source, module_id, scope_class=None, scope_func=None)

    def _walk_body(
        self,
        body: Iterable[ast.stmt],
        source: str,
        module_id: str,
        scope_class: Optional[str],
        scope_func: Optional[str],
    ) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_id = self._register_function(
                    node, source, module_id, scope_class, scope_func
                )
                self._walk_body(
                    node.body, source, module_id,
                    scope_class=scope_class, scope_func=func_id,
                )
            elif isinstance(node, ast.ClassDef):
                class_id = self._register_class(node, source, module_id, scope_class)
                self._walk_body(
                    node.body, source, module_id,
                    scope_class=class_id, scope_func=None,
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._record_import(node, module_id)
            else:
                self._walk_node(node, module_id, scope_class, scope_func)

    def _walk_node(self, node, module_id, scope_class, scope_func) -> None:
        scope_key = scope_func or module_id
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                raw = _call_name(child.func)
                if raw:
                    attr = raw.rsplit(".", 1)[-1]
                    if "." in raw:
                        self._maybe_dynamic_names.add(attr)
                    self._unresolved_calls.append(
                        UnresolvedCall(src_id=scope_key, raw_name=raw,
                                       attr=attr, lineno=child.lineno))
            elif isinstance(child, ast.Assign):
                self._track_assignment(child, scope_key, scope_class)
            elif isinstance(child, ast.AnnAssign):
                self._track_ann_assignment(child, scope_key, scope_class)

    
    def _register_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        module_id: str,
        scope_class: Optional[str],
        scope_func: Optional[str],
    ) -> str:
        parent_id = scope_class or module_id
        parent_sym = self.graph.symbols[parent_id]
        if scope_class:
            qual = f"{parent_sym.qualified_name}.{node.name}"
            kind = NodeKind.METHOD
        elif scope_func:
            parent_func = self.graph.symbols[scope_func]
            qual = f"{parent_func.qualified_name}.{node.name}"
            kind = NodeKind.FUNCTION
        else:
            qual = f"{parent_sym.qualified_name}.{node.name}"
            kind = NodeKind.FUNCTION

        sym_id = f"{module_id}::{_qualname_for(module_id, qual)}"
        sym = Symbol(
            id=sym_id, name=node.name, qualified_name=qual, kind=kind,
            file=module_id, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            col_offset=node.col_offset, parent=parent_id,
            signature=_signature(node),
            docstring=ast.get_docstring(node),
            decorators=[_decorator_str(d) for d in node.decorator_list],
            source=_source_snippet(node, source),
        )
        self.graph.add_symbol(sym)
        self.graph.add_edge(parent_id, sym_id, EdgeKind.CONTAINS)

        # Track annotated parameters for later type inference
        scope = {}
        for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
            if arg.annotation is not None:
                t = _annotation_name(arg.annotation)
                if t and t != "None":
                    scope[arg.arg] = t
        for vararg in (node.args.vararg, node.args.kwarg):
            if vararg is not None and vararg.annotation is not None:
                t = _annotation_name(vararg.annotation)
                if t and t != "None":
                    scope[vararg.arg] = t
        if scope:
            self._scope_vars[sym_id] = scope

        return sym_id

    def _register_class(
        self,
        node: ast.ClassDef,
        source: str,
        module_id: str,
        scope_class: Optional[str],
    ) -> str:
        parent_id = scope_class or module_id
        parent_sym = self.graph.symbols[parent_id]
        qual = f"{parent_sym.qualified_name}.{node.name}"
        sym_id = f"{module_id}::{_qualname_for(module_id, qual)}"
        sym = Symbol(
            id=sym_id, name=node.name, qualified_name=qual, kind=NodeKind.CLASS,
            file=module_id, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            col_offset=node.col_offset, parent=parent_id,
            docstring=ast.get_docstring(node),
            decorators=[_decorator_str(d) for d in node.decorator_list],
            source=_source_snippet(node, source),
            metadata={"bases": [ast.unparse(b) for b in node.bases]},
        )
        self.graph.add_symbol(sym)
        self.graph.add_edge(parent_id, sym_id, EdgeKind.CONTAINS)
        return sym_id

    # --- imports ---

    def _record_import(self, node: ast.Import | ast.ImportFrom, module_id: str) -> None:
        bucket = self._imports_by_module[module_id]
        if isinstance(node, ast.Import):
            for alias in node.names:
                bucket.append({
                    "type": "import", "module": alias.name,
                    "alias": alias.asname, "level": 0,
                })
        else:
            for alias in node.names:
                bucket.append({
                    "type": "from", "module": node.module or "",
                    "name": alias.name, "alias": alias.asname,
                    "level": node.level,
                })

    # --- phase 3a: resolve imports ---

    def _track_assignment(self, node, scope_key, scope_class) -> None:
        if not hasattr(self, "_pending_assignments"):
            self._pending_assignments = []
        if not hasattr(self, "_scope_vars"):
            self._scope_vars = {}
        if not hasattr(self, "_class_attrs"):
            self._class_attrs = {}
        if not hasattr(node, "targets") or len(node.targets) != 1:
            return
        tgt = node.targets[0]
        value = node.value
        if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self" and scope_class):
            cls = _extract_constructor_call(value)
            if cls:
                self._class_attrs.setdefault(scope_class, {})[tgt.attr] = cls
            return
        if not isinstance(tgt, ast.Name):
            return
        cls = _extract_constructor_call(value)
        if cls:
            self._scope_vars.setdefault(scope_key, {})[tgt.id] = cls
            return
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            self._pending_assignments.append((scope_key, tgt.id, value.func.id))

    def _track_ann_assignment(self, node, scope_key, scope_class) -> None:
        if not hasattr(self, "_scope_vars"):
            self._scope_vars = {}
        if not hasattr(self, "_class_attrs"):
            self._class_attrs = {}
        t = _annotation_name(node.annotation)
        if not t or t == "None":
            return
        tgt = node.target
        if isinstance(tgt, ast.Name):
            self._scope_vars.setdefault(scope_key, {})[tgt.id] = t
        elif (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
              and tgt.value.id == "self" and scope_class):
            self._class_attrs.setdefault(scope_class, {})[tgt.attr] = t
    def _resolve_pending_assignments(self) -> None:
        pending = getattr(self, "_pending_assignments", [])
        if not pending:
            return
        by_name = {}
        for sym in self.graph.symbols.values():
            if sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                by_name.setdefault(sym.name, []).append(sym)
        scope_vars = self.__dict__.setdefault("_scope_vars", {})
        return_types = getattr(self, "_return_types", {})
        for scope_key, var_name, func_name in pending:
            candidates = by_name.get(func_name, [])
            if len(candidates) != 1:
                continue
            ret_type = return_types.get(candidates[0].id)
            if not ret_type:
                ret_type = _extract_return_type_from_sig(candidates[0].signature)
            if ret_type:
                scope_vars.setdefault(scope_key, {})[var_name] = ret_type

    def _resolve_imports(self) -> None:
        for module_id, imports in self._imports_by_module.items():
            for imp in imports:
                target = self._resolve_module_name(
                    imp["module"], imp.get("level", 0), module_id
                )
                if target and target in self.graph.symbols:
                    self.graph.add_edge(module_id, target, EdgeKind.IMPORTS)

    def _resolve_module_name(
        self, dotted: str, level: int, from_module_id: str
    ) -> Optional[str]:
        if level > 0:
            pkg_parts = from_module_id.split("/")[:-level]
            if dotted:
                pkg_parts += dotted.split(".")
            dotted = ".".join(p for p in pkg_parts if p)
        if not dotted:
            return None
        if dotted in self._module_id_by_dotted:
            return self._module_id_by_dotted[dotted]
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in self._module_id_by_dotted:
                return self._module_id_by_dotted[candidate]
        return None

    # --- phase 3b: resolve calls ---

    def _resolve_calls(self) -> None:
        by_name: dict[str, list[Symbol]] = {}
        for sym in self.graph.symbols.values():
            if sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                by_name.setdefault(sym.name, []).append(sym)
        for call in self._unresolved_calls:
            if call.src_id not in self.graph.symbols:
                continue
            target = self._resolve_one_call(call, by_name)
            if target:
                self.graph.add_edge(call.src_id, target, EdgeKind.CALLS)

    def _resolve_one_call(
        self, call: UnresolvedCall, by_name: dict[str, list[Symbol]]
    ) -> Optional[str]:
        src_sym = self.graph.symbols.get(call.src_id)
        if src_sym is None:
            return None

        # Never resolve stdlib-prefixed calls — kills ast.parse→Parser.parse
        if call.raw_name.startswith(_STDLIB_PREFIXES):
            return None

        parts = call.raw_name.split(".")

        # A. self.method() / cls.method()
        if len(parts) == 2 and parts[0] in ("self", "cls"):
            parent = src_sym.parent
            if parent and parent in self.graph.symbols:
                for sym in self.graph.symbols.values():
                    if sym.parent == parent and sym.name == parts[1] \
                            and sym.kind == NodeKind.METHOD:
                        return sym.id
            return None

        # B. self.attr.method() — resolve via tracked attribute type
        if len(parts) == 3 and parts[0] in ("self", "cls"):
            parent = src_sym.parent
            if parent:
                attr_type = getattr(self, '_class_attrs', {}).get(parent, {}).get(parts[1])
                if attr_type:
                    class_sym = self._find_class_by_name(attr_type, src_sym.file)
                    if class_sym:
                        m = self._find_method(class_sym.id, parts[2])
                        if m:
                            return m
            return None

        # C. x.method() where x is a tracked local var or annotated param
        if len(parts) == 2:
            var, method = parts
            scope_vars = getattr(self, '_scope_vars', {}).get(call.src_id, {})
            if var in scope_vars:
                class_sym = self._find_class_by_name(scope_vars[var], src_sym.file)
                if class_sym:
                    m = self._find_method(class_sym.id, method)
                    if m:
                        return m

        # D. Bare name → module-level function in same file
        if "." not in call.raw_name:
            for sym in self.graph.symbols.values():
                if sym.file == src_sym.file and sym.name == call.attr \
                        and sym.kind == NodeKind.FUNCTION \
                        and sym.parent == src_sym.file:
                    return sym.id

        # D. ClassName.method() - chained constructor
        if len(parts) == 2 and parts[0] and parts[0][0].isupper():
            class_sym = self._find_class_by_name(parts[0], src_sym.file)
            if class_sym:
                m = self._find_method(class_sym.id, parts[1])
                if m:
                    return m

        # E. Bare name → sibling method in enclosing class
        if "." not in call.raw_name:
            parent = src_sym.parent
            if parent and parent in self.graph.symbols:
                for sym in self.graph.symbols.values():
                    if sym.parent == parent and sym.name == call.attr \
                            and sym.kind == NodeKind.METHOD:
                        return sym.id

        # F. Imported names
        for imp in self._imports_by_module.get(src_sym.file, []):
            target_mod = self._resolve_module_name(
                imp.get("module", ""), imp.get("level", 0), src_sym.file
            )
            if not target_mod:
                continue
            for sym in self.graph.symbols.values():
                if sym.file == target_mod and sym.name == call.attr:
                    return sym.id

        # G. Unique global fallback (bare names only)
        if "." not in call.raw_name:
            candidates = by_name.get(call.attr, [])
            if len(candidates) == 1:
                return candidates[0].id

        return None

    def _find_class_by_name(self, name: str, current_file: str) -> Optional[Symbol]:
        # Same file
        for sym in self.graph.symbols.values():
            if sym.kind == NodeKind.CLASS and sym.name == name \
                    and sym.file == current_file:
                return sym
        # Imported
        for imp in self._imports_by_module.get(current_file, []):
            target_mod = self._resolve_module_name(
                imp.get("module", ""), imp.get("level", 0), current_file
            )
            if not target_mod:
                continue
            for sym in self.graph.symbols.values():
                if sym.kind == NodeKind.CLASS and sym.name == name \
                        and sym.file == target_mod:
                    return sym
        # Global unique
        matches = [s for s in self.graph.symbols.values()
                   if s.kind == NodeKind.CLASS and s.name == name]
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_method(self, class_id: str, method_name: str) -> Optional[str]:
        for sym in self.graph.symbols.values():
            if sym.parent == class_id and sym.name == method_name \
                    and sym.kind == NodeKind.METHOD:
                return sym.id
        return None


# =============================================================================
# Parser Helpers
# =============================================================================


def _path_to_dotted(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1] or ["__init__"]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _qualname_for(module_id: str, qual: str) -> str:
    mod_dotted = _path_to_dotted(module_id)
    if qual.startswith(mod_dotted + "."):
        return qual[len(mod_dotted) + 1:]
    return qual


def _extract_return_type_from_sig(sig):
    """Extract return type from a signature string like 'def f() -> Foo'."""
    if not sig or "->" not in sig:
        return None
    rt = sig.split("->", 1)[1].strip()
    if "[" in rt:
        rt = rt.split("[")[0].strip()
    rt = rt.rstrip(":")
    if not rt or rt == "None" or not rt[0].isalpha():
        return None
    return rt


def _annotation_name(node: Optional[ast.expr]) -> Optional[str]:
    """Best-effort simple type name from an annotation AST node."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.slice)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            if not (isinstance(side, ast.Constant) and side.value is None):
                return _annotation_name(side)
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    return None


def _extract_constructor_call(node: ast.expr) -> Optional[str]:
    """Return 'ClassName' if node is ClassName(...) with a simple name."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name and name[0].isupper():
                return name
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name and name[0].isupper():
                return name
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = _fmt_arguments(node.args)
    ret = ""
    if node.returns is not None:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return f"{prefix} {node.name}({args}){ret}"


def _fmt_arguments(args: ast.arguments) -> str:
    parts: list[str] = []
    posonly = args.posonlyargs
    posargs = args.args
    defaults = args.defaults
    all_pos = posonly + posargs
    offset = len(all_pos) - len(defaults)

    def render(arg: ast.arg, default: Optional[ast.expr]) -> str:
        s = arg.arg
        if arg.annotation is not None:
            try:
                s += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        if default is not None:
            try:
                s += f" = {ast.unparse(default)}"
            except Exception:
                s += " = ..."
        return s

    for i, a in enumerate(posonly):
        d = defaults[i - offset] if i >= offset else None
        parts.append(render(a, d))
    if posonly:
        parts.append("/")
    for i, a in enumerate(posargs):
        idx = i + len(posonly)
        d = defaults[idx - offset] if idx >= offset else None
        parts.append(render(a, d))
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(render(a, d))
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return ", ".join(parts)


def _decorator_str(node: ast.expr) -> str:
    try:
        return "@" + ast.unparse(node)
    except Exception:
        return "@<expr>"


def _source_snippet(node: ast.AST, source: str) -> str:
    seg = ast.get_source_segment(source, node)
    if seg:
        return seg
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])


def _call_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return None


# =============================================================================
# Symbol Resolution
# =============================================================================


def find_symbols(graph: CodeGraph, query: str) -> list[Symbol]:
    q = query.strip()
    if q in graph.symbols:
        return [graph.symbols[q]]
    module_hits = [
        s for s in graph.symbols.values()
        if s.kind == NodeKind.MODULE and (s.file == q or s.file.endswith("/" + q))
    ]
    if module_hits:
        return module_hits
    qual_hits: list[Symbol] = []
    for s in graph.symbols.values():
        if s.qualified_name == q:
            qual_hits.append(s)
        elif s.qualified_name.endswith("." + q):
            qual_hits.append(s)
    if qual_hits:
        return sorted(qual_hits, key=lambda s: (len(s.qualified_name), s.file, s.lineno))
    short_hits = [s for s in graph.symbols.values() if s.name == q]
    return sorted(short_hits, key=lambda s: (s.file, s.lineno))


def pick_symbol(graph: CodeGraph, query: str) -> Optional[Symbol]:
    hits = find_symbols(graph, query)
    if not hits:
        _err(f"No symbol matched: {query!r}")
        _hint("try `codeatlas list` or `codeatlas search <substring>`")
        return None
    if len(hits) == 1:
        return hits[0]
    _render_candidates(hits, title=f"multiple matches for {query!r}")
    _hint(f"rerun with the full qualified name, e.g. "
          f"{_console_style(hits[0].qualified_name, 'cyan')}")
    return None


# =============================================================================
# Graph Analytics
# =============================================================================


def compute_impact(graph: CodeGraph, target_id: str, max_depth: int = 20) -> dict[str, int]:
    depths: dict[str, int] = {target_id: 0}
    frontier = [target_id]
    for depth in range(1, max_depth + 1):
        next_frontier: list[str] = []
        for sid in frontier:
            for caller in graph.callers_of(sid):
                if caller.id not in depths:
                    depths[caller.id] = depth
                    next_frontier.append(caller.id)
        if not next_frontier:
            break
        frontier = next_frontier
    return depths


def _recurse_tree(graph: CodeGraph, root_id: str, max_depth: int, direction: str) -> dict:
    def recurse(sid: str, remaining: int, path: frozenset[str]) -> dict:
        sym = graph.symbols.get(sid)
        node = {"sym": sym, "children": [], "cycle": False}
        if remaining <= 0 or sym is None:
            return node
        neighbors = (graph.callers_of(sid) if direction == "callers"
                     else graph.callees_of(sid))
        for n in neighbors:
            if n.id in path:
                node["children"].append({"sym": n, "children": [], "cycle": True})
            else:
                node["children"].append(recurse(n.id, remaining - 1, path | {sid}))
        return node
    return recurse(root_id, max_depth, frozenset())


def impact_tree(graph: CodeGraph, root_id: str, max_depth: int = 5) -> dict:
    return _recurse_tree(graph, root_id, max_depth, "callers")


def callee_tree(graph: CodeGraph, root_id: str, max_depth: int = 5) -> dict:
    return _recurse_tree(graph, root_id, max_depth, "callees")


def find_hotspots(graph: CodeGraph, top: int = 15) -> list[tuple[Symbol, int]]:
    ranked = [(s, len(graph.callers_of(s.id)))
              for s in graph.symbols.values()
              if s.kind in (NodeKind.FUNCTION, NodeKind.METHOD)]
    ranked.sort(key=lambda x: (-x[1], x[0].file, x[0].lineno))
    return ranked[:top]


_DUNDER_OR_ENTRY = {
    "main", "__init__", "__main__", "__new__", "__call__",
    "__enter__", "__exit__", "setUp", "tearDown",
    "app", "cli", "run", "entry", "console_main",
}


def find_dead_code(graph: CodeGraph) -> list[Symbol]:
    dead: list[Symbol] = []
    for s in graph.symbols.values():
        if s.kind not in (NodeKind.FUNCTION, NodeKind.METHOD):
            continue
        if s.name in _DUNDER_OR_ENTRY or s.name.startswith(("__", "test_", "Test", "setup_", "teardown_")):
            continue
        if s.decorators and any("property" in d or "staticmethod" in d
                                or "classmethod" in d for d in s.decorators):
            continue
        if s.name in graph.maybe_dynamic_calls:
            continue
        if not graph.callers_of(s.id):
            dead.append(s)
    dead.sort(key=lambda s: (s.file, s.lineno))
    return dead


def find_cycles(graph: CodeGraph, min_size: int = 2) -> list[list[str]]:
    adj: dict[str, list[str]] = {
        sid: [c.id for c in graph.callees_of(sid)] for sid in graph.symbols
    }
    counter = [0]
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = counter[0]
        lowlinks[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) >= min_size:
                sccs.append(scc)

    sys.setrecursionlimit(max(10000, len(adj) * 2 + 100))
    for v in adj:
        if v not in indices:
            strongconnect(v)
    return sccs


# =============================================================================
# Retrieval for `ask` — hybrid graph expansion + keyword scoring
# =============================================================================


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "how", "what", "where", "when", "why", "who", "which", "does", "do",
    "did", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "not", "no", "but", "if", "then", "else", "this", "that",
    "it", "its", "as", "so", "up", "out", "can", "could", "should", "would",
    "will", "we", "us", "you", "i", "me", "my", "our", "your",
    "explain", "describe", "show", "tell", "work", "works", "working",
}


@dataclass
class RetrievalHit:
    symbol: Symbol
    reason: str
    score: int = 0


def retrieve_context(
    graph: CodeGraph, question: str, max_seeds: int = 6, max_total: int = 14
) -> list[RetrievalHit]:
    """Hybrid retrieval: keyword-scored seeds + graph-neighbor expansion."""
    words = [
        w for w in (question.lower().replace("?", " ").replace(",", " ")
                    .replace(".", " ").split())
        if len(w) > 2 and w not in _STOPWORDS
    ]
    if not words:
        return []

    scored: list[tuple[int, Symbol]] = []
    for sym in graph.symbols.values():
        if sym.kind == NodeKind.MODULE:
            continue
        name_l = sym.name.lower()
        qual_l = sym.qualified_name.lower()
        doc_l = (sym.docstring or "").lower()
        score = 0
        for w in words:
            if w in name_l:
                score += 6
            if w in qual_l:
                score += 2
            if w in doc_l:
                score += 3
        if score > 0:
            scored.append((score, sym))

    scored.sort(key=lambda x: (-x[0], x[1].file, x[1].lineno))
    seeds = scored[:max_seeds]

    hits: list[RetrievalHit] = []
    seen: set[str] = set()
    for score, sym in seeds:
        if sym.id not in seen:
            hits.append(RetrievalHit(sym, f"matched question (score={score})", score))
            seen.add(sym.id)

    # Graph expansion — one hop out from each seed
    for _, seed in seeds:
        for neighbor in graph.callees_of(seed.id) + graph.callers_of(seed.id):
            if neighbor.id in seen:
                continue
            if neighbor.kind == NodeKind.MODULE:
                continue
            direction = ("callee" if neighbor in graph.callees_of(seed.id)
                         else "caller")
            hits.append(RetrievalHit(
                neighbor, f"{direction} of {seed.name}", 0))
            seen.add(neighbor.id)
            if len(hits) >= max_total:
                break
        if len(hits) >= max_total:
            break

    return hits[:max_total]


def build_ask_prompt(question: str, hits: list[RetrievalHit]) -> str:
    lines = [
        "You are an expert Python code analyst. Answer the user's question using "
        "ONLY the code context provided below.",
        "",
        "Rules:",
        "- Cite every claim with `file:line` format (e.g., 'codeatlas.py:252').",
        "- If the context is insufficient, say so explicitly — do not guess.",
        "- Be concise. Aim for 3-8 sentences, plus code snippets when helpful.",
        "",
        f"QUESTION: {question}",
        "",
        "CODE CONTEXT:",
    ]
    for hit in hits:
        sym = hit.symbol
        lines.append("")
        lines.append(f"### {sym.qualified_name}  ({sym.file}:{sym.lineno})")
        if sym.signature:
            lines.append(f"Signature: {sym.signature}")
        if sym.docstring:
            doc = sym.docstring.strip().split("\n")
            doc = "\n".join(doc[:8])  # trim long docstrings
            lines.append(f"Docstring: {doc}")
        if sym.source:
            src = sym.source
            if len(src) > 1800:
                src = src[:1800] + "\n# ... (truncated)"
            lines.append("Source:")
            lines.append(src)

    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


# =============================================================================
# LLM Backends — pure stdlib (urllib), zero extra deps
# =============================================================================


def _detect_backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=2
        ) as r:
            if r.status == 200:
                return "ollama"
    except Exception:
        pass
    raise RuntimeError(
        "No LLM backend available.\n"
        "  Options:\n"
        "    1. Install Ollama (free, offline):  winget install Ollama.Ollama\n"
        "    2. Set ANTHROPIC_API_KEY in your environment\n"
        "    3. Set OPENAI_API_KEY in your environment\n"
        "    4. Use --dry-run to see retrieved context without an LLM"
    )


def call_llm(prompt: str, backend: Optional[str] = None,
             model: Optional[str] = None) -> str:
    backend = backend or _detect_backend()
    if backend == "ollama":
        return _call_ollama(prompt, model or "llama3.2")
    if backend == "anthropic":
        return _call_anthropic(prompt, model or "claude-haiku-4-5")
    if backend == "openai":
        return _call_openai(prompt, model or "gpt-4o-mini")
    raise ValueError(f"unknown backend: {backend}")


def _call_ollama(prompt: str, model: str) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at localhost:11434 — {e}")


def _call_anthropic(prompt: str, model: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read())
        return data["content"][0]["text"].strip()


def _call_openai(prompt: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()


# =============================================================================
# Rendering
# =============================================================================


def _console_style(text: str, style: str) -> str:
    if RICH_AVAILABLE:
        return f"[{style}]{text}[/]"
    return text


def _err(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[bold red]✗[/] {msg}")
    else:
        print(f"✗ {msg}", file=sys.stderr)


def _hint(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[dim]{msg}[/]")
    else:
        print(msg, file=sys.stderr)


def _require_rich() -> bool:
    if RICH_AVAILABLE:
        return True
    _err("This command needs `rich` for pretty output. Install it with:")
    print("    pip install rich", file=sys.stderr)
    return False


def _emit_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _render_candidates(symbols: list[Symbol], title: str) -> None:
    if not symbols:
        _hint(f"(no {title})")
        return
    if not RICH_AVAILABLE:
        print(f"\n{title}:")
        for s in symbols:
            print(f"  {s.qualified_name}  ({s.file}:{s.lineno})")
        return
    table = Table(
        title=title, box=rich_box.SIMPLE_HEAD, show_lines=False,
        title_justify="left", title_style="bold", expand=False,
    )
    table.add_column("qualified name", style="cyan")
    table.add_column("kind", style="dim")
    table.add_column("location", style="dim")
    for s in symbols:
        table.add_row(s.qualified_name, s.kind.value, f"{s.file}:{s.lineno}")
    console.print(table)


def render_symbol(graph: CodeGraph, sym: Symbol) -> None:
    if not RICH_AVAILABLE:
        _render_symbol_plain(graph, sym)
        return
    title = Text()
    title.append(sym.qualified_name, style="bold cyan")
    title.append("   ")
    title.append(f"{sym.kind.value}", style="yellow")
    title.append("  ·  ")
    title.append(f"{sym.file}:{sym.lineno}", style="dim")
    console.print(Panel(title, box=rich_box.ROUNDED, border_style="cyan",
                        padding=(0, 1)))
    console.print()

    if sym.signature:
        console.print("[bold yellow]Signature[/]")
        console.print(Syntax(sym.signature, "python", theme="monokai",
                             word_wrap=True, background_color="default"))
        console.print()
    if sym.docstring:
        console.print("[bold yellow]Docstring[/]")
        console.print(f"[italic]{sym.docstring.strip()}[/]")
        console.print()
    if sym.source:
        console.print("[bold yellow]Source[/]")
        console.print(Syntax(
            sym.source, "python", theme="monokai",
            line_numbers=True, start_line=sym.lineno,
            word_wrap=True, background_color="default",
        ))
        console.print()

    children = graph.children_of(sym.id)
    if children:
        console.print(f"[bold blue]Contains[/] [dim]({len(children)})[/]")
        for c in sorted(children, key=lambda x: x.lineno):
            console.print(f"  [blue]·[/] {c.kind.value:<8} "
                          f"[cyan]{c.name}[/]  [dim]{c.file}:{c.lineno}[/]")
        console.print()

    callees = sorted(graph.callees_of(sym.id), key=lambda s: s.qualified_name)
    if callees:
        console.print(f"[bold green]Calls →[/] [dim]({len(callees)})[/]")
        for c in callees:
            console.print(f"  [green]→[/] [cyan]{c.qualified_name}[/]  "
                          f"[dim]{c.file}:{c.lineno}[/]")
        console.print()

    callers = sorted(graph.callers_of(sym.id), key=lambda s: s.qualified_name)
    if callers:
        console.print(f"[bold magenta]Called by ←[/] [dim]({len(callers)})[/]")
        for c in callers:
            console.print(f"  [magenta]←[/] [cyan]{c.qualified_name}[/]  "
                          f"[dim]{c.file}:{c.lineno}[/]")
        console.print()


def _render_symbol_plain(graph: CodeGraph, sym: Symbol) -> None:
    print(f"\n=== {sym.qualified_name} ({sym.kind.value}) ===")
    print(f"at {sym.file}:{sym.lineno}")
    if sym.signature:
        print(f"\nSignature:\n  {sym.signature}")
    if sym.docstring:
        print(f"\nDocstring:\n  {sym.docstring.strip()}")
    if sym.source:
        print(f"\nSource:\n{sym.source}")
    for label, neighbors in (
        ("Contains", graph.children_of(sym.id)),
        ("Calls", graph.callees_of(sym.id)),
        ("Called by", graph.callers_of(sym.id)),
    ):
        if neighbors:
            print(f"\n{label} ({len(neighbors)}):")
            for n in sorted(neighbors, key=lambda x: x.qualified_name):
                print(f"  {n.qualified_name}  ({n.file}:{n.lineno})")


def _build_rich_tree(node: dict, root: bool = False) -> "Tree":
    sym: Symbol = node["sym"]
    if sym is None:
        return Tree("[red]<unknown>[/]")
    label = Text()
    if root:
        label.append("◉ ", style="bold cyan")
    label.append(sym.qualified_name, style="cyan" if root else "white")
    label.append("   ")
    label.append(f"{sym.file}:{sym.lineno}", style="dim")
    if node.get("cycle"):
        label.append("  (cycle)", style="yellow")
    t = Tree(label)
    for child in node["children"]:
        t.add(_build_rich_tree(child, root=False))
    return t


def _tree_to_json(node: dict) -> dict:
    sym: Optional[Symbol] = node.get("sym")
    return {
        "symbol": sym.qualified_name if sym else None,
        "location": f"{sym.file}:{sym.lineno}" if sym else None,
        "kind": sym.kind.value if sym else None,
        "cycle": node.get("cycle", False),
        "children": [_tree_to_json(c) for c in node.get("children", [])],
    }


def _symbol_to_json(sym: Symbol) -> dict:
    return {
        "symbol": sym.qualified_name,
        "kind": sym.kind.value,
        "location": f"{sym.file}:{sym.lineno}",
        "signature": sym.signature,
    }


# =============================================================================
# Load / Index / Stats
# =============================================================================


def _is_stale(target: Path, graph_path: Path) -> bool:
    try:
        graph_mtime = graph_path.stat().st_mtime
    except OSError:
        return True
    for py_file in target.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        try:
            if py_file.stat().st_mtime > graph_mtime:
                return True
        except OSError:
            continue
    return False


def _load_or_build(target: Path, verbose: bool = False) -> CodeGraph:
    graph_path = target / GRAPH_DIR / GRAPH_FILE
    if graph_path.exists() and not _is_stale(target, graph_path):
        try:
            return CodeGraph.load(graph_path)
        except Exception as e:
            if verbose:
                print(f"[codeatlas] cache load failed ({e}), rebuilding",
                      file=sys.stderr)
    elif graph_path.exists() and verbose:
        print("[codeatlas] cache is stale, rebuilding", file=sys.stderr)
    return Parser(target, verbose=verbose).parse()


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _summary(graph: CodeGraph, elapsed: float) -> str:
    counts = {k: 0 for k in NodeKind}
    for s in graph.symbols.values():
        counts[s.kind] += 1
    ec = {k: 0 for k in EdgeKind}
    for e in graph.edges:
        ec[e.kind] += 1
    return "\n".join([
        f"Root:        {graph.root}",
        f"Files:       {len({s.file for s in graph.symbols.values()})}",
        f"Symbols:     {len(graph.symbols)}  "
        f"({counts[NodeKind.CLASS]} classes, "
        f"{counts[NodeKind.FUNCTION]} functions, "
        f"{counts[NodeKind.METHOD]} methods)",
        f"Edges:       {len(graph.edges)}  "
        f"({ec[EdgeKind.CALLS]} calls, "
        f"{ec[EdgeKind.IMPORTS]} imports, "
        f"{ec[EdgeKind.INHERITS]} inherits)",
        f"Parsed in:   {elapsed * 1000:.0f} ms",
        f"Graph size:  {_human_bytes(len(json.dumps(graph.to_dict())))}",
    ])


def cmd_index(target: str, verbose: bool = False, json_out: bool = False) -> int:
    root = Path(target)
    t0 = time.perf_counter()
    graph = Parser(root, verbose=verbose).parse()
    elapsed = time.perf_counter() - t0
    out = root / GRAPH_DIR / GRAPH_FILE
    graph.save(out)
    if json_out:
        _emit_json({
            "root": graph.root, "symbols": len(graph.symbols),
            "edges": len(graph.edges),
            "elapsed_ms": round(elapsed * 1000, 1),
            "saved_to": str(out),
        })
        return 0
    print(_summary(graph, elapsed))
    print(f"Saved:       {out}")
    return 0


def cmd_stats(target: str, verbose: bool = False, json_out: bool = False) -> int:
    root = Path(target)
    t0 = time.perf_counter()
    graph = Parser(root, verbose=verbose).parse()
    elapsed = time.perf_counter() - t0
    if json_out:
        counts = {k.value: 0 for k in NodeKind}
        for s in graph.symbols.values():
            counts[s.kind.value] += 1
        ec = {k.value: 0 for k in EdgeKind}
        for e in graph.edges:
            ec[e.kind.value] += 1
        _emit_json({
            "root": graph.root,
            "files": len({s.file for s in graph.symbols.values()}),
            "symbols": len(graph.symbols),
            "symbols_by_kind": counts,
            "edges": len(graph.edges),
            "edges_by_kind": ec,
            "elapsed_ms": round(elapsed * 1000, 1),
        })
        return 0
    print(_summary(graph, elapsed))
    return 0


# =============================================================================
# Inspection Commands
# =============================================================================


def cmd_explain(target: str, query: str, verbose: bool = False,
                json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    sym = pick_symbol(graph, query)
    if sym is None:
        return 1
    if json_out:
        _emit_json({
            "symbol": sym.qualified_name, "kind": sym.kind.value,
            "location": f"{sym.file}:{sym.lineno}",
            "signature": sym.signature, "docstring": sym.docstring,
            "source": sym.source,
            "contains": [_symbol_to_json(c) for c in graph.children_of(sym.id)],
            "calls": [_symbol_to_json(c) for c in graph.callees_of(sym.id)],
            "called_by": [_symbol_to_json(c) for c in graph.callers_of(sym.id)],
        })
        return 0
    if not _require_rich():
        return 1
    render_symbol(graph, sym)
    return 0


def cmd_tree(target: str, query: str, depth: int = 4, verbose: bool = False,
             json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    sym = pick_symbol(graph, query)
    if sym is None:
        return 1
    tree_data = callee_tree(graph, sym.id, max_depth=depth)
    if json_out:
        _emit_json(_tree_to_json(tree_data))
        return 0
    if not _require_rich():
        return 1
    console.print(Panel(
        f"[bold]callee tree[/] · [cyan]{sym.qualified_name}[/] · depth={depth}",
        box=rich_box.ROUNDED, border_style="green",
    ))
    console.print(_build_rich_tree(tree_data, root=True))
    return 0


def cmd_impact(target: str, query: str, depth: int = 4, verbose: bool = False,
               json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    sym = pick_symbol(graph, query)
    if sym is None:
        return 1
    depths = compute_impact(graph, sym.id)
    impacted = {sid: d for sid, d in depths.items() if d > 0}
    if json_out:
        _emit_json({
            "target": sym.qualified_name,
            "direct": sum(1 for d in impacted.values() if d == 1),
            "transitive": len(impacted),
            "max_depth": max(depths.values()) if depths else 0,
            "tree": _tree_to_json(impact_tree(graph, sym.id, max_depth=depth)),
        })
        return 0
    if not _require_rich():
        return 1
    if not impacted:
        console.print(Panel(
            f"[green]No internal callers of [cyan]{sym.qualified_name}[/] "
            f"— it's a leaf or entry point.",
            border_style="green", box=rich_box.ROUNDED,
        ))
        return 0
    by_depth: dict[int, list[str]] = {}
    for sid, d in impacted.items():
        by_depth.setdefault(d, []).append(sid)
    stats = Text()
    stats.append("Target:    ", style="bold")
    stats.append(f"{sym.qualified_name}\n", style="cyan")
    stats.append("Direct:    ", style="bold")
    stats.append(f"{len(by_depth.get(1, []))}\n", style="yellow")
    stats.append("Transitive:", style="bold")
    stats.append(f"{len(impacted)}\n", style="red")
    stats.append("Max depth: ", style="bold")
    stats.append(f"{max(depths.values())} hop(s)", style="dim")
    console.print(Panel(stats, title="impact", border_style="red",
                        box=rich_box.ROUNDED, padding=(0, 1)))
    console.print()
    console.print(_build_rich_tree(
        impact_tree(graph, sym.id, max_depth=depth), root=True))
    return 0


def cmd_hotspots(target: str, top: int = 15, verbose: bool = False,
                 json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    ranked = find_hotspots(graph, top=top)
    if json_out:
        _emit_json([{"symbol": s.qualified_name, "kind": s.kind.value,
                     "location": f"{s.file}:{s.lineno}", "callers": c}
                    for s, c in ranked])
        return 0
    if not _require_rich():
        return 1
    if not ranked:
        _hint("(no callable symbols found)")
        return 0
    table = Table(
        title=f"top {len(ranked)} hotspots  ·  ranked by caller count",
        box=rich_box.ROUNDED, title_justify="left", title_style="bold",
        border_style="cyan",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("callers", justify="right", style="bold yellow")
    table.add_column("symbol", style="cyan")
    table.add_column("location", style="dim")
    for i, (sym, count) in enumerate(ranked, 1):
        table.add_row(str(i), f"{count:>3} " + "▇" * min(count, 20),
                      sym.qualified_name, f"{sym.file}:{sym.lineno}")
    console.print(table)
    return 0


def cmd_dead_code(target: str, verbose: bool = False,
                  json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    dead = find_dead_code(graph)
    if json_out:
        _emit_json([{"symbol": s.qualified_name, "kind": s.kind.value,
                     "location": f"{s.file}:{s.lineno}"} for s in dead])
        return 0
    if not _require_rich():
        return 1
    if not dead:
        console.print(Panel(
            "[green]✓ No dead code detected — every internal function "
            "is called somewhere.[/]",
            border_style="green", box=rich_box.ROUNDED,
        ))
        return 0
    table = Table(
        title=f"potential dead code  ·  {len(dead)} function(s) with 0 callers",
        box=rich_box.ROUNDED, title_justify="left", title_style="bold",
        border_style="yellow",
    )
    table.add_column("symbol", style="cyan")
    table.add_column("kind", style="dim")
    table.add_column("location", style="dim")
    for sym in dead:
        table.add_row(sym.qualified_name, sym.kind.value,
                      f"{sym.file}:{sym.lineno}")
    console.print(table)
    _hint("excludes dunders, entry points, property/class/static methods, "
          "and names reached via attribute access")
    return 0


def cmd_list(target: str, kind: Optional[str] = None, verbose: bool = False,
             json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    filter_kind = None
    if kind:
        try:
            filter_kind = NodeKind(kind)
        except ValueError:
            _err(f"unknown kind {kind!r}. Valid: {[k.value for k in NodeKind]}")
            return 1
    syms = list(graph.symbols.values())
    if filter_kind:
        syms = [s for s in syms if s.kind == filter_kind]
    syms.sort(key=lambda s: (s.file, s.lineno))
    if json_out:
        _emit_json([_symbol_to_json(s) for s in syms])
        return 0
    if not _require_rich():
        return 1
    _render_candidates(syms, title=f"{len(syms)} symbol(s)" +
                       (f" of kind {filter_kind.value}" if filter_kind else ""))
    return 0


def cmd_search(target: str, pattern: str, verbose: bool = False,
               json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    pat = pattern.lower()
    hits = [s for s in graph.symbols.values()
            if pat in s.name.lower() or pat in s.qualified_name.lower()]
    hits.sort(key=lambda s: (s.file, s.lineno))
    if json_out:
        _emit_json([_symbol_to_json(s) for s in hits])
        return 0
    if not _require_rich():
        return 1
    _render_candidates(hits, title=f"matches for {pattern!r}  ·  {len(hits)} hit(s)")
    return 0


def cmd_callers(target: str, query: str, verbose: bool = False,
                json_out: bool = False) -> int:
    return _one_hop(target, query, "callers", verbose, json_out)


def cmd_callees(target: str, query: str, verbose: bool = False,
                json_out: bool = False) -> int:
    return _one_hop(target, query, "callees", verbose, json_out)


def _one_hop(target: str, query: str, direction: str, verbose: bool,
             json_out: bool) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    sym = pick_symbol(graph, query)
    if sym is None:
        return 1
    neighbors = (graph.callers_of(sym.id) if direction == "callers"
                 else graph.callees_of(sym.id))
    if json_out:
        _emit_json({
            "target": sym.qualified_name, "direction": direction,
            "count": len(neighbors),
            "symbols": [_symbol_to_json(n) for n in neighbors],
        })
        return 0
    if not _require_rich():
        return 1
    arrow = "←" if direction == "callers" else "→"
    header = Text()
    header.append(sym.qualified_name, style="bold cyan")
    header.append(f"  {arrow}  ")
    header.append(f"{len(neighbors)} {direction}", style="yellow")
    console.print(Panel(header, box=rich_box.ROUNDED, border_style="cyan"))
    if not neighbors:
        _hint(f"(none — this {sym.kind.value} has no {direction} inside the repo)")
        return 0
    for n in sorted(neighbors, key=lambda x: x.qualified_name):
        console.print(f"  [green]{arrow}[/] [cyan]{n.qualified_name}[/]  "
                      f"[dim]{n.file}:{n.lineno}[/]")
    return 0


def cmd_cycles(target: str, verbose: bool = False, json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    cycles = find_cycles(graph, min_size=2)
    if json_out:
        _emit_json([{
            "size": len(c),
            "symbols": [{"symbol": graph.symbols[s].qualified_name,
                         "location": f"{graph.symbols[s].file}:{graph.symbols[s].lineno}"}
                        for s in c],
        } for c in cycles])
        return 0
    if not _require_rich():
        return 1
    if not cycles:
        console.print(Panel("[green]✓ No call cycles detected.[/]",
                            border_style="green", box=rich_box.ROUNDED))
        return 0
    console.print(Panel(f"[bold red]Found {len(cycles)} cycle(s) of size ≥ 2[/]",
                        border_style="red", box=rich_box.ROUNDED))
    for i, cyc in enumerate(cycles, 1):
        console.print(f"\n[bold yellow]Cycle {i}[/] [dim]({len(cyc)} symbols)[/]")
        for sid in cyc:
            sym = graph.symbols[sid]
            console.print(f"  [red]◉[/] [cyan]{sym.qualified_name}[/]  "
                          f"[dim]{sym.file}:{sym.lineno}[/]")
    return 0


def cmd_graph(target: str, symbol: Optional[str] = None, depth: int = 2,
              verbose: bool = False, json_out: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    if symbol:
        root = pick_symbol(graph, symbol)
        if root is None:
            return 1
        ids = {root.id}
        frontier = [root.id]
        for _ in range(depth):
            nxt: list[str] = []
            for sid in frontier:
                for c in graph.callees_of(sid) + graph.callers_of(sid):
                    if c.id not in ids:
                        ids.add(c.id)
                        nxt.append(c.id)
            frontier = nxt
        edges = [e for e in graph.edges
                 if e.kind == EdgeKind.CALLS and e.src in ids and e.dst in ids]
        if json_out:
            _emit_json({
                "nodes": [{"id": graph.symbols[s].qualified_name,
                           "kind": graph.symbols[s].kind.value} for s in ids],
                "edges": [{"src": graph.symbols[e.src].qualified_name,
                           "dst": graph.symbols[e.dst].qualified_name,
                           "kind": e.kind.value} for e in edges],
            })
            return 0
        lines = ["flowchart LR"]
        for sid in sorted(ids, key=lambda x: graph.symbols[x].qualified_name):
            sym = graph.symbols[sid]
            lines.append(f'    {_mermaid_id(sid)}["{_mermaid_label(sym.qualified_name)}"]')
        for e in edges:
            lines.append(f"    {_mermaid_id(e.src)} --> {_mermaid_id(e.dst)}")
        print("\n".join(lines))
        return 0

    module_ids = [s.id for s in graph.symbols.values() if s.kind == NodeKind.MODULE]
    import_edges = [e for e in graph.edges if e.kind == EdgeKind.IMPORTS]
    if json_out:
        _emit_json({
            "nodes": [{"id": graph.symbols[mid].file, "kind": "module"}
                      for mid in module_ids],
            "edges": [{"src": graph.symbols[e.src].file,
                       "dst": graph.symbols[e.dst].file,
                       "kind": "imports"} for e in import_edges],
        })
        return 0
    lines = ["flowchart LR"]
    for mid in module_ids:
        lines.append(f'    {_mermaid_id(mid)}["{_mermaid_label(graph.symbols[mid].file)}"]')
    for e in import_edges:
        lines.append(f"    {_mermaid_id(e.src)} -->|imports| {_mermaid_id(e.dst)}")
    print("\n".join(lines))
    return 0


def _mermaid_id(sym_id: str) -> str:
    return "n" + "".join(c if c.isalnum() else "_" for c in sym_id)


def _mermaid_label(s: str) -> str:
    return s.replace('"', "'").replace("[", "(").replace("]", ")")


# =============================================================================
# Ask Command
# =============================================================================


def _render_context_summary(hits: list[RetrievalHit], show_source: bool = False) -> None:
    if not RICH_AVAILABLE:
        print(f"\nRetrieved {len(hits)} symbols:")
        for h in hits:
            print(f"  · {h.symbol.qualified_name}  ({h.reason})")
        if show_source:
            for h in hits:
                print(f"\n### {h.symbol.qualified_name}")
                print(h.symbol.source)
        return
    if not hits:
        _hint("(no relevant symbols found)")
        return
    table = Table(
        title=f"retrieved context · {len(hits)} symbol(s)",
        box=rich_box.ROUNDED, title_justify="left", title_style="bold",
        border_style="blue",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("symbol", style="cyan")
    table.add_column("location", style="dim")
    table.add_column("why", style="dim")
    for i, h in enumerate(hits, 1):
        table.add_row(str(i), h.symbol.qualified_name,
                      f"{h.symbol.file}:{h.symbol.lineno}", h.reason)
    console.print(table)

    if show_source:
        for h in hits:
            console.print()
            console.print(Panel(
                Syntax(h.symbol.source or "", "python", theme="monokai",
                       line_numbers=True, start_line=h.symbol.lineno,
                       word_wrap=True, background_color="default"),
                title=f"[bold]{h.symbol.qualified_name}[/]",
                border_style="dim",
            ))


def _render_answer(answer: str, hits: list[RetrievalHit]) -> None:
    if not RICH_AVAILABLE:
        print("\n" + answer)
        return
    console.print()
    console.print(Panel(
        Markdown(answer),
        title="[bold cyan]answer[/]",
        border_style="cyan",
        padding=(1, 2),
    ))
    if hits:
        console.print()
        _hint(f"grounded in {len(hits)} symbol(s) — re-run with --why to see them")


def cmd_ask(target: str, question: str, backend: Optional[str] = None,
            model: Optional[str] = None, dry_run: bool = False,
            why: bool = False, json_out: bool = False,
            verbose: bool = False) -> int:
    graph = _load_or_build(Path(target), verbose=verbose)
    hits = retrieve_context(graph, question)

    if json_out:
        result = {
            "question": question,
            "context": [
                {"symbol": h.symbol.qualified_name,
                 "kind": h.symbol.kind.value,
                 "location": f"{h.symbol.file}:{h.symbol.lineno}",
                 "reason": h.reason} for h in hits
            ],
            "answer": None,
        }
        if not dry_run:
            try:
                prompt = build_ask_prompt(question, hits)
                result["answer"] = call_llm(prompt, backend=backend, model=model)
            except Exception as e:
                result["error"] = str(e)
        _emit_json(result)
        return 0

    if not _require_rich():
        return 1

    if why or dry_run:
        _render_context_summary(hits, show_source=dry_run)

    if dry_run:
        _hint("(dry run — no LLM called. Remove --dry-run to generate an answer.)")
        return 0

    if not hits:
        _err("No relevant code found for that question. Try different keywords.")
        return 1

    try:
        resolved = backend or _detect_backend()
    except RuntimeError as e:
        _err(str(e))
        _hint("Showing retrieved context instead:")
        _render_context_summary(hits, show_source=False)
        return 1

    prompt = build_ask_prompt(question, hits)

    with console.status(f"[cyan]asking {resolved}...[/]", spinner="dots"):
        try:
            answer = call_llm(prompt, backend=resolved, model=model)
        except Exception as e:
            _err(f"LLM call failed: {e}")
            return 1

    _render_answer(answer, hits)
    return 0


# =============================================================================
# Argument Parsing
# =============================================================================


def _extract_flag(args: list[str], flag: str) -> Optional[str]:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def _extract_path(args: list[str]) -> str:
    return _extract_flag(args, "--path") or "."


_FLAGS_WITH_VALUE = {
    "--path", "--depth", "--top", "--kind", "--symbol", "--backend", "--model",
}


def _positional(args: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in _FLAGS_WITH_VALUE:
            i += 2
            continue
        if a.startswith("--"):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _parse_int(args: list[str], flag: str, default: int) -> Optional[int]:
    val = _extract_flag(args, flag)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        _err(f"{flag} must be an integer, got {val!r}")
        return None


# =============================================================================
# main()
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    cmd, *rest = argv
    verbose = "--verbose" in rest or "-v" in rest
    json_out = "--json" in rest
    rest = [a for a in rest if a not in {"--verbose", "-v", "--json"}]

    if cmd == "index":
        pos = _positional(rest)
        return cmd_index(pos[0] if pos else ".", verbose=verbose, json_out=json_out)
    if cmd == "stats":
        pos = _positional(rest)
        return cmd_stats(pos[0] if pos else ".", verbose=verbose, json_out=json_out)

    if cmd == "explain":
        pos = _positional(rest)
        if not pos:
            _err("usage: codeatlas explain <symbol> [--path <repo>] [--json]")
            return 2
        return cmd_explain(_extract_path(rest), pos[0], verbose=verbose,
                           json_out=json_out)
    if cmd == "tree":
        pos = _positional(rest)
        if not pos:
            _err("usage: codeatlas tree <symbol> [--depth N] [--json]")
            return 2
        depth = _parse_int(rest, "--depth", 4)
        if depth is None:
            return 2
        return cmd_tree(_extract_path(rest), pos[0], depth=depth,
                        verbose=verbose, json_out=json_out)
    if cmd == "impact":
        pos = _positional(rest)
        if not pos:
            _err("usage: codeatlas impact <symbol> [--depth N] [--json]")
            return 2
        depth = _parse_int(rest, "--depth", 4)
        if depth is None:
            return 2
        return cmd_impact(_extract_path(rest), pos[0], depth=depth,
                          verbose=verbose, json_out=json_out)
    if cmd == "list":
        return cmd_list(_extract_path(rest), kind=_extract_flag(rest, "--kind"),
                        verbose=verbose, json_out=json_out)
    if cmd == "search":
        pos = _positional(rest)
        if not pos:
            _err("usage: codeatlas search <pattern> [--json]")
            return 2
        return cmd_search(_extract_path(rest), pos[0], verbose=verbose,
                          json_out=json_out)
    if cmd == "callers":
        pos = _positional(rest)
        if not pos:
            _err("usage: codeatlas callers <symbol> [--json]")
            return 2
        return cmd_callers(_extract_path(rest), pos[0], verbose=verbose,
                           json_out=json_out)
    if cmd == "callees":
        pos = _positional(rest)
        if not pos:
            _err("usage: codeatlas callees <symbol> [--json]")
            return 2
        return cmd_callees(_extract_path(rest), pos[0], verbose=verbose,
                           json_out=json_out)
    if cmd == "hotspots":
        top = _parse_int(rest, "--top", 15)
        if top is None:
            return 2
        return cmd_hotspots(_extract_path(rest), top=top, verbose=verbose,
                            json_out=json_out)
    if cmd == "dead-code":
        return cmd_dead_code(_extract_path(rest), verbose=verbose,
                             json_out=json_out)
    if cmd == "cycles":
        return cmd_cycles(_extract_path(rest), verbose=verbose,
                          json_out=json_out)
    if cmd == "graph":
        depth = _parse_int(rest, "--depth", 2)
        if depth is None:
            return 2
        return cmd_graph(_extract_path(rest),
                         symbol=_extract_flag(rest, "--symbol"),
                         depth=depth, verbose=verbose, json_out=json_out)
    if cmd == "ask":
        pos = _positional(rest)
        if not pos:
            _err('usage: codeatlas ask "<question>" [--backend X] [--dry-run]')
            return 2
        question = " ".join(pos)
        return cmd_ask(
            _extract_path(rest), question,
            backend=_extract_flag(rest, "--backend"),
            model=_extract_flag(rest, "--model"),
            dry_run="--dry-run" in rest,
            why="--why" in rest,
            json_out=json_out,
            verbose=verbose,
        )
    if cmd == "version":
        print(f"codeatlas {__version__}")
        return 0

    _err(f"unknown command: {cmd}")
    print(__doc__, file=sys.stderr)
    return 2


def app():
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())