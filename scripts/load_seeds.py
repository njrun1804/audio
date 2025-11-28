#!/usr/bin/env python3
"""Load all seed data into the ASR dictionary."""

from pathlib import Path
import sys


def main():
    from asr.dictionary import init_db, import_from_json
    from asr.dictionary.db import get_stats

    seeds_dir = Path(__file__).parent.parent / "seeds"

    print("Initializing database...")
    init_db()

    total = 0
    for tier_file in sorted(seeds_dir.glob("tier_*.json")):
        count = import_from_json(tier_file)
        print(f"  Loaded {tier_file.name}: {count} entries")
        total += count

    # Also copy context profiles to the right place
    contexts_src = seeds_dir / "contexts"
    if contexts_src.exists():
        from asr.dictionary.selector import BiasListSelector
        selector = BiasListSelector()
        for ctx_file in contexts_src.glob("*.json"):
            import json
            with open(ctx_file) as f:
                data = json.load(f)
            from asr.dictionary.models import ContextProfile
            profile = ContextProfile(**data)
            selector.save_context_profile(profile)
            print(f"  Loaded context: {profile.name}")

    stats = get_stats()
    print(f"\nTotal: {stats.total_entries} entries loaded")
    print(f"Tiers: {dict(stats.entries_by_tier)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
