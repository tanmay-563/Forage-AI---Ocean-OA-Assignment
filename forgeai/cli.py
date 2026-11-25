from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from .pipeline import build_agent
from .runner import from_definition
from .tools import build_registry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="forgeai", description="Build and run evidence-grounded document agents")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Generate an agent, evaluate it, optimize, and report")
    build.add_argument("--brief", required=True, help="Natural-language brief or @path to a UTF-8 brief file")
    build.add_argument("--documents", required=True, type=Path)
    build.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    build.add_argument("--interactive", action="store_true", help="Ask generated intake questions instead of assuming conservative defaults")
    build.add_argument("--iterations", type=int, choices=range(4), default=3)
    build.add_argument("--top-k", type=int, choices=range(1, 13), default=5, help="Initial retrieval candidate count; optimizer may raise it after retrieval misses")
    chat = commands.add_parser("chat", help="Run any generated agent definition")
    chat.add_argument("--agent", required=True, type=Path)
    tools = commands.add_parser("tools", help="List registered tools")
    tools.add_argument("--documents", type=Path)
    tool = commands.add_parser("tool", help="Execute one registered tool with JSON arguments")
    tool.add_argument("--name", required=True)
    tool.add_argument("--args", default="{}", help="JSON object of validated tool arguments")
    tool.add_argument("--documents", type=Path)
    evaluate = commands.add_parser("evaluate", help="Run a saved evaluation suite")
    evaluate.add_argument("--agent", required=True, type=Path)
    evaluate.add_argument("--cases", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("FORGEAI_LOG_LEVEL", "INFO").upper(), format="%(levelname)s %(name)s: %(message)s")

    if args.command == "build":
        brief = Path(args.brief[1:]).read_text(encoding="utf-8") if args.brief.startswith("@") else args.brief
        clarifications = []
        if args.interactive:
            from .intake import intake
            _, questions = intake(brief, args.documents, unattended=False)
            for question in questions:
                clarifications.append(input(f"{question}\n> ").strip())
        try:
            report = build_agent(brief, args.documents, args.artifacts, unattended=not args.interactive, clarifications=clarifications, iterations=args.iterations, top_k=args.top_k)
        except Exception as exc:
            logging.getLogger("forgeai").error("Build failed (%s): %s", type(exc).__name__, exc)
            return 1
        print(json.dumps({"agent_id": report["agent_id"], "artifact_directory": str((args.artifacts / report["agent_id"]).resolve()), "evaluation_cases": report["evaluation"]["case_count"], "final_pass_rate": report["final_score"]["pass_rate"]}, indent=2))
        return 0
    if args.command == "chat":
        runner = from_definition(args.agent)
        print(f"{runner.definition.name} ({runner.provider.name}). Type /quit to exit.")
        while True:
            try: question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt): print(); break
            if question.lower() in {"/quit", "/exit"}: break
            try:
                output = runner.answer(question)
                print("agent> " + output["answer"])
                if output["sources"]: print("sources> " + "; ".join(f"{s['source']}" + (f" row {s['row']}" if s["row"] else "") for s in output["sources"]))
            except Exception as exc: print(f"error> {exc}")
        return 0
    if args.command == "tools":
        print(json.dumps(build_registry(args.documents).list(), indent=2)); return 0
    if args.command == "tool":
        registry = build_registry(args.documents)
        print(json.dumps(registry.execute(args.name, json.loads(args.args)), indent=2)); return 0
    if args.command == "evaluate":
        from .pipeline import load_cases
        from .evaluation.optimizer import run_suite
        from .evaluation.judge import score
        runner = from_definition(args.agent)
        results = run_suite(runner, load_cases(args.cases))
        print(json.dumps(score(results), indent=2)); return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
