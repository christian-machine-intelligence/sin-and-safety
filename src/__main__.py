"""CLI entry point: python -m src <command>"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src {generate|evaluate|analyze|compare|pilot}")
        print()
        print("Commands:")
        print("  generate   Phase 1 — build the 700-act benchmark with gpt-5.4")
        print("  evaluate   Phase 2 — multi-condition judgments (Claude or GPT)")
        print("  analyze    Phase 3 — contingency, McNemar, per-sin/subject gap, figures")
        print("  compare    Cross-model comparison from per-model summary_stats")
        print("  pilot      End-to-end smoke test on 14 acts (2 per sin)")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip command from argv

    if command == "generate":
        from .generate_benchmark import main as run
        run()
    elif command == "evaluate":
        from .evaluate import main as run
        run()
    elif command in ("analyze", "analysis"):
        from .analysis import main as run
        run()
    elif command == "compare":
        from .compare import main as run
        run()
    elif command == "radar":
        from .radar import main as run
        run()
    elif command == "robustness":
        from .robustness import main as run
        run()
    elif command in ("validate-tags", "validate_tags"):
        from .validate_tags import main as run
        run()
    elif command == "pilot":
        from .pilot import main as run
        run()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
