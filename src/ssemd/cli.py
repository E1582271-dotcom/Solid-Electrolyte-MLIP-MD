"""Command line: ``ssemd classify | tables | figures | docs | check-repo | all``."""
from __future__ import annotations

import argparse
import sys


def _classify(_args) -> int:
    from . import classify, data

    rows = classify.build()
    data.classification.cache_clear()
    counts = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    print("classified", len(rows), counts)
    return 0


def _tables(_args) -> int:
    from . import tables

    tables.build()
    print("wrote data/workflow.csv")
    return 0


def _figures(args) -> int:
    from .figures import registry

    for name in registry.build(args.only):
        print("wrote", name)
    return 0


def _docs(_args) -> int:
    from . import report

    for path in report.build():
        print("wrote", path)
    return 0


def _check(_args) -> int:
    from . import hygiene

    problems = hygiene.check()
    for problem in problems:
        print(problem)
    print("repository checks:", "FAIL" if problems else "pass")
    return 1 if problems else 0


def _all(args) -> int:
    for step in (_classify, _tables, _figures, _docs):
        step(args)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ssemd", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("classify", help="classify the 62 compositions from data/runs.csv").set_defaults(func=_classify)
    sub.add_parser("tables", help="write data/workflow.csv").set_defaults(func=_tables)
    figures = sub.add_parser("figures", help="draw every figure from the committed tables")
    figures.add_argument("--only", nargs="*", default=None, help="figure names, e.g. fig1 ed02")
    figures.set_defaults(func=_figures)
    sub.add_parser("docs", help="write the generated sections of the documentation").set_defaults(func=_docs)
    sub.add_parser("check-repo", help="no CJK text, no private identifiers, file-size limits").set_defaults(func=_check)
    everything = sub.add_parser("all", help="classify, tables, figures and docs")
    everything.add_argument("--only", nargs="*", default=None)
    everything.set_defaults(func=_all)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
