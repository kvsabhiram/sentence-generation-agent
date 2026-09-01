"""CLI entrypoint for the synthetic sentence generation pipeline."""

from __future__ import annotations

import argparse

from pipeline.orchestrator import replay_from_raw, run, run_synchronous


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic sentence generation pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the pipeline config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "Reprocess whatever generator output already exists in the raw log "
            "(merging every word's candidates across all prior generation attempts) "
            "and judge it, without calling Gemini for anything new. Words still "
            "short of their quota afterward are left pending for a normal run later."
        ),
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Run using plain synchronous API calls (a thread pool of workers) instead "
            "of the async Batch API. Use this when a provider's Batch API is "
            "unavailable (e.g. an outage) -- costs full price (no batch discount) but "
            "doesn't depend on batch job infrastructure at all."
        ),
    )
    args = parser.parse_args()
    if args.replay:
        replay_from_raw(config_path=args.config)
    elif args.sync:
        run_synchronous(config_path=args.config)
    else:
        run(config_path=args.config)


if __name__ == "__main__":
    main()
