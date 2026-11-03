#!/usr/bin/env python3
"""Inventory Warp source without importing either package or running kernels.

This is a source survey, not an accumulator inference engine. The companion
assessments.json contains the manually reviewed neighborhood semantics. Exact
assignments, reductions, returns and stores remain available for every decorated
definition, including helpers outside wp_ filenames and nested kernel factories.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


DECORATORS = {"wp.func", "wp.kernel", "wp.func_grad", "wp.func_replay"}


def expression(node: ast.AST | None) -> str:
    return ast.unparse(node) if node is not None else ""


def root_expression(node: ast.AST) -> str:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return expression(node)


def decorator_name(node: ast.AST) -> str:
    return expression(node.func if isinstance(node, ast.Call) else node)


def revision(repository: Path) -> dict:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments], text=True
        ).strip()

    return {
        "root": str(repository),
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
    }


def scan_function(node: ast.FunctionDef, package: str, path: str, sha: str) -> dict:
    descendants = list(ast.walk(node))
    arguments = [
        {"name": item.arg, "annotation": expression(item.annotation)}
        for item in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    ]
    updates, writes, recurrences = [], [], []
    for item in descendants:
        if isinstance(item, ast.AugAssign):
            updates.append({
                "line": item.lineno, "target": expression(item.target),
                "operator": type(item.op).__name__, "value": expression(item.value),
            })
        targets = item.targets if isinstance(item, ast.Assign) else (
            [item.target] if isinstance(item, (ast.AnnAssign, ast.AugAssign)) else []
        )
        for target in targets:
            if isinstance(target, ast.Subscript):
                writes.append({
                    "line": item.lineno, "target": expression(target),
                    "root": root_expression(target), "value": expression(item.value),
                    "operator": type(item.op).__name__ if isinstance(item, ast.AugAssign) else "Assign",
                })
            if isinstance(item, ast.Assign) and isinstance(target, ast.Name):
                if any(isinstance(value, ast.Name) and value.id == target.id for value in ast.walk(item.value)):
                    recurrences.append({
                        "line": item.lineno, "target": target.id,
                        "value": expression(item.value),
                    })
    calls = [item for item in descendants if isinstance(item, ast.Call)]
    reductions = [
        {"line": item.lineno, "call": expression(item)}
        for item in calls
        if any(word in expression(item.func).lower() for word in (
            "atomic_", "tile_sum", "tile_reduce", "tile_max", "tile_min",
            "lanesum", "lanemax", "lanemin", "reduce",
        ))
    ]
    return {
        "id": f"{package}:{path}:{node.lineno}",
        "package": package, "path": path, "name": node.name,
        "decorators": [expression(item) for item in node.decorator_list],
        "kind": next(decorator_name(item) for item in node.decorator_list
                     if decorator_name(item) in DECORATORS),
        "line": node.lineno, "end_line": node.end_lineno,
        "arguments": arguments,
        "loops": [
            {"line": item.lineno, "target": expression(item.target), "range": expression(item.iter)}
            for item in descendants if isinstance(item, ast.For)
        ],
        "while_loops": [
            {"line": item.lineno, "condition": expression(item.test)}
            for item in descendants if isinstance(item, ast.While)
        ],
        "augmented_assignments": updates,
        "self_reassignments": recurrences,
        "indexed_writes": writes,
        "returns": [
            {"line": item.lineno, "value": expression(item.value)}
            for item in descendants if isinstance(item, ast.Return)
        ],
        "calls": sorted({expression(item.func) for item in calls}),
        "reduction_calls": reductions,
        "indexed_loads": sorted({
            expression(item) for item in descendants
            if isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Load)
        }),
        "source_sha256": sha,
        "definition_ast_sha256": hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest(),
    }


def scan(package: str, repository: Path) -> tuple[list, list]:
    root = repository / "src" / ("warpSPHCore" if package == "core" else "warpSPH")
    files, functions = [], []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        definitions = [item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)]
        decorated = [
            item for item in definitions
            if any(decorator_name(value) in DECORATORS for value in item.decorator_list)
        ]
        if not path.name.startswith("wp_") and not decorated:
            continue
        relative = str(path.relative_to(repository))
        sha = hashlib.sha256(source.encode()).hexdigest()
        functions.extend(scan_function(item, package, relative, sha) for item in decorated)
        files.append({
            "id": f"{package}:{relative}", "package": package, "path": relative,
            "wp_filename": path.name.startswith("wp_"), "source_sha256": sha,
            "decorated_definitions": len(decorated),
            "python_definitions": [
                {"line": item.lineno, "name": item.name}
                for item in definitions if item not in decorated
            ],
            "imports": [expression(item) for item in tree.body
                        if isinstance(item, (ast.Import, ast.ImportFrom))],
        })
    return files, functions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--core-reference", type=Path,
                        help="Read-only comparison with another core checkout, accounting for moved primal files")
    parser.add_argument("--assessments", type=Path,
                        default=Path(__file__).with_name("assessments.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {"core": args.core.resolve(), "frontend": args.frontend.resolve()}
    manual = json.loads(args.assessments.read_text())
    files, functions = [], []
    for package, repository in roots.items():
        package_files, package_functions = scan(package, repository)
        files.extend(package_files)
        functions.extend(package_functions)
    assessments = manual["files"]
    actual = {item["id"] for item in files}
    if actual != set(assessments):
        raise SystemExit(f"Assessment coverage mismatch: missing={sorted(actual-set(assessments))}, stale={sorted(set(assessments)-actual)}")
    ids = {item["id"] for item in functions}
    assert len(ids) == len(functions), "Duplicate source definition IDs"
    for item in files:
        if item["source_sha256"] != assessments[item["id"]]["reviewed_source_sha256"]:
            raise SystemExit(f"Source changed since semantic review: {item['id']}")
        item["assessment"] = assessments[item["id"]]
    for item in functions:
        assessment = assessments[f"{item['package']}:{item['path']}"]
        item["file_family"] = assessment["family"]
        item["semantic_review"] = "file assessment plus exact local syntax; no automatic scalar count inference"
    for operation in manual["operations"]:
        assert operation["package"] + ":" + operation["path"] in actual
        available = [item for item in functions
                     if item["package"] == operation["package"] and item["path"] == operation["path"]]
        for anchor in operation["anchors"]:
            assert any(item["name"] == anchor for item in available), (operation["name"], anchor)
    # Link launch and adjacency definitions to the semantic passes through local
    # calls. Keep helpers distinguishable from the neighborhood state they feed.
    links = {item["id"]: set() for item in functions}
    for operation in manual["operations"]:
        for item in functions:
            if (item["package"] == operation["package"] and item["path"] == operation["path"]
                    and item["name"] in operation["anchors"]):
                links[item["id"]].add(operation["name"])
    for _ in range(len(functions)):
        changed = False
        for item in functions:
            for called in functions:
                if (called["package"] == item["package"] and called["path"] == item["path"]
                        and called["name"] in item["calls"]):
                    old = len(links[item["id"]])
                    links[item["id"]].update(links[called["id"]])
                    changed |= len(links[item["id"]]) != old
        if not changed:
            break
    for item in functions:
        item["operation_links"] = sorted(links[item["id"]])
        if item["kind"] == "wp.func_grad":
            role = "custom local adjoint"
        elif item["kind"] == "wp.kernel":
            role = "kernel entry; inspect linked pass or file assessment for output scope"
        elif "Adjacency" in item["name"]:
            role = "adjacency/range orchestration"
        elif "Func_i" in item["name"] or item["name"] in {"_sums", "_materialDensitySums", "_countNeighborsCompactHashMapRow", "countNeighborsVerletFunc", "updateNeighborsVerletFunc"}:
            role = "range/candidate accumulation or edge emission"
        else:
            role = "local math, pair evaluation, loading or indexing helper"
        item["source_role"] = role
    counts = {}
    for package in roots:
        package_files = [item for item in files if item["package"] == package]
        package_functions = [item for item in functions if item["package"] == package]
        counts[package] = {
            "files": len(package_files),
            "wp_files": sum(item["wp_filename"] for item in package_files),
            "decorated_files": sum(item["decorated_definitions"] > 0 for item in package_files),
            "definitions": len(package_functions),
            "definitions_in_wp_files": sum(Path(item["path"]).name.startswith("wp_") for item in package_functions),
            "decorator_counts": dict(Counter(item["kind"] for item in package_functions)),
        }
    output = {
        "schema_version": 1,
        "method": "Static AST inventory and manual file/operation assessments; no imports or kernel executions. Local indexed writes include by-value vectors/matrices and are not all global stores. Source definitions are counted once, independent of overload instantiation and factory specialization. Generated Warp adjoints are not separate source definitions.",
        "repositories": {key: revision(value) for key, value in roots.items()},
        "counts": counts,
        "operation_counts": dict(Counter(item["package"] for item in manual["operations"])),
        "operation_categories": dict(Counter(item["category"] for item in manual["operations"])),
        "files": files, "operations": manual["operations"], "functions": functions,
    }
    if args.core_reference:
        reference = args.core_reference.resolve()
        reference_files, reference_functions = scan("core", reference)
        def canonical(path: str) -> str:
            return path.replace("coreOperations/native/wp_", "coreOperations/wp_")
        primary = {
            (canonical(item["path"]), item["name"], item["kind"], item["definition_ast_sha256"])
            for item in functions if item["package"] == "core"
        }
        secondary = {
            (canonical(item["path"]), item["name"], item["kind"], item["definition_ast_sha256"])
            for item in reference_functions
        }
        differing_paths = sorted({item[0] for item in primary ^ secondary})
        output["core_reference"] = {
            **revision(reference), "files": len(reference_files),
            "definitions": len(reference_functions),
            "comparison": "Decorated AST hashes after canonicalizing coreOperations/native/wp_ relocations; aliases have no definitions.",
            "different_definition_paths": differing_paths,
            "reference_definitions_at_different_paths": [
                item for item in reference_functions if canonical(item["path"]) in differing_paths
            ],
            "source_files": reference_files,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"counts": counts, "operation_counts": output["operation_counts"],
                      "operation_categories": output["operation_categories"]}, indent=2))


if __name__ == "__main__":
    main()
