import sys
import importlib

COMMAND_MAPPING = {
    "headroom": "highway.benchmarks.ragbench_headroom_like",
    "ministress": "highway.benchmarks.ragbench_ministress",
    "scaleup": "highway.benchmarks.ragbench_scaleup",
    "llm-runtime-fake": "highway.benchmarks.llm_runtime_fake",
    "local-llm-quality": "highway.benchmarks.local_llm_quality",
    "long-conversation-quality": "highway.benchmarks.long_conversation_quality",
    "multi-theme-long-llm": "highway.benchmarks.multi_theme_long_llm",
    "ooc-scaleup": "highway.benchmarks.ooc_scaleup",
    "poc234-kernel-hardening": "highway.runners.run_poc234_kernel_hardening",
    "quality-token-tradeoff": "highway.benchmarks.quality_token_tradeoff",
    "runtime-perf-margin": "highway.benchmarks.runtime_perf_margin",
    "semantic-ann-quality": "highway.benchmarks.semantic_ann_quality",
    "swebench-contextpack": "highway.benchmarks.swebench_contextpack",
    "token-economics-smoke": "highway.benchmarks.token_economics_smoke",
    "build-hardening-workload": "highway.workloads.build_poc234_kernel_hardening_workload",
    "eval-pccc": "highway.benchmarks.eval_pccc_benchmark",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: highway <command> [args]")
        print("\nAvailable commands:")
        for cmd in sorted(COMMAND_MAPPING.keys()):
            print(f"  {cmd}")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMAND_MAPPING:
        print(f"Error: Unknown command '{cmd}'", file=sys.stderr)
        print("\nAvailable commands:", file=sys.stderr)
        for c in sorted(COMMAND_MAPPING.keys()):
            print(f"  {c}", file=sys.stderr)
        sys.exit(1)

    module_name = COMMAND_MAPPING[cmd]
    # Shift sys.argv to remove the subcommand
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Import and run main
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"Error: Module '{module_name}' does not have a main() function.", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
