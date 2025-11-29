#!/usr/bin/env python3
"""Load all seed data into the ASR dictionary."""

from pathlib import Path
import sys
import json


def main():
    try:
        from asr.dictionary import init_db, import_from_json
        from asr.dictionary.db import get_stats
        from asr.dictionary.selector import BiasListSelector
        from asr.dictionary.models import ContextProfile
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}", file=sys.stderr)
        return 1

    seeds_dir = Path(__file__).parent.parent / "seeds"

    # Validate seeds directory exists
    if not seeds_dir.exists():
        print(f"ERROR: Seeds directory not found: {seeds_dir}", file=sys.stderr)
        return 1

    print("Initializing database...")
    try:
        init_db()
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}", file=sys.stderr)
        return 1

    total = 0
    tier_files = sorted(seeds_dir.glob("tier_*.json"))
    if not tier_files:
        print("WARNING: No tier_*.json files found in seeds directory", file=sys.stderr)

    for tier_file in tier_files:
        try:
            count = import_from_json(tier_file)
            print(f"  Loaded {tier_file.name}: {count} entries")
            total += count
        except Exception as e:
            print(f"ERROR: Failed to load {tier_file.name}: {e}", file=sys.stderr)
            return 1

    # Also copy context profiles to the right place
    contexts_src = seeds_dir / "contexts"
    if contexts_src.exists():
        selector = BiasListSelector()
        ctx_files = list(contexts_src.glob("*.json"))

        for ctx_file in ctx_files:
            try:
                with open(ctx_file) as f:
                    data = json.load(f)
                profile = ContextProfile(**data)
                selector.save_context_profile(profile)
                print(f"  Loaded context: {profile.name}")
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON in {ctx_file.name}: {e}", file=sys.stderr)
                return 1
            except Exception as e:
                print(f"ERROR: Failed to load context {ctx_file.name}: {e}", file=sys.stderr)
                return 1

    try:
        stats = get_stats()
        print(f"\nTotal: {stats.total_entries} entries loaded")
        print(f"Tiers: {dict(stats.entries_by_tier)}")
    except Exception as e:
        print(f"ERROR: Failed to get stats: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
