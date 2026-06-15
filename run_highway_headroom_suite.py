from highway.benchmarks.ragbench_headroom_like import main

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc
