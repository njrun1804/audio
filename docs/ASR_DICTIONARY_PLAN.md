# ASR Dictionary System - Implementation Plan

## Executive Summary

Transform the current flat vocabulary files into a rich, context-aware proper noun knowledge base with:
- **Tiered vocabularies** (always-on core + domain packs)
- **Rich entry model** (aliases, pronunciations, boosting weights, source tracking)
- **Context-aware selection** (top-k ranked bias lists per session)
- **Multi-layer integration** (Whisper prompts + Claude correction + feedback loop)

---

## 1. Current State Analysis

### What Exists
```
src/asr/vocabularies/
├── common_names.py      # 500 static names (first/last/historians)
└── tech_terms.py        # Static tech companies/products/terms

~/.asr/vocabularies/
└── {domain}.txt         # One term per line, no metadata
```

### Gaps
| Missing Feature | Impact |
|----------------|--------|
| No aliases | "Navesink 12K" vs "Navesink Challenge" are separate |
| No pronunciations | Can't guide Whisper on "Chernow" vs "Sharno" |
| No boosting weights | All terms equal priority |
| No type classification | Can't filter by person/product/location |
| No context tags | Can't scope to meeting/project/domain |
| No source tracking | Don't know where terms came from |
| No tier system | 10k+ flat list swamps the model |

---

## 2. Target Architecture

### 2.1 Data Model

```
~/.asr/dictionaries/
├── dictionary.db          # SQLite: entries, aliases, pronunciations
├── contexts/              # Context packs (JSON)
│   ├── running.json
│   ├── veeva.json
│   └── asr_dev.json
└── sources/               # Source snapshots for debugging
    └── contacts_2024-11-28.json
```

### 2.2 Entry Schema (SQLite)

```sql
-- Core entries table
CREATE TABLE entries (
    id TEXT PRIMARY KEY,           -- UUID
    canonical TEXT NOT NULL,       -- "Navesink Challenge"
    display TEXT,                  -- "Navesink Challenge 12K"
    type TEXT CHECK(type IN ('person', 'org', 'product', 'event',
                              'location', 'jargon', 'misc')),
    language TEXT DEFAULT 'en',
    tier TEXT CHECK(tier IN ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')),
    boost_weight REAL DEFAULT 1.0, -- 0.0-3.0 (like AWS Lex)
    occurrence_count INTEGER DEFAULT 0,
    last_seen_at TEXT,             -- ISO timestamp
    source TEXT,                   -- "contacts", "calendar", "manual"
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Aliases for each entry
CREATE TABLE aliases (
    id INTEGER PRIMARY KEY,
    entry_id TEXT REFERENCES entries(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,           -- "Navesink", "Navesink 12k"
    is_common_misspelling BOOLEAN DEFAULT FALSE,
    UNIQUE(entry_id, alias)
);

-- Pronunciations (optional, for future G2P integration)
CREATE TABLE pronunciations (
    id INTEGER PRIMARY KEY,
    entry_id TEXT REFERENCES entries(id) ON DELETE CASCADE,
    ipa TEXT,                      -- /næˈvɛsɪŋk/
    phoneme_sequence TEXT,         -- "N AE V EH S IH NG K"
    language TEXT DEFAULT 'en',    -- Pronunciation language
    variant TEXT                   -- "american", "british", etc.
);

-- Context tags for filtering
CREATE TABLE entry_contexts (
    entry_id TEXT REFERENCES entries(id) ON DELETE CASCADE,
    context TEXT NOT NULL,         -- "running", "veeva", "family"
    PRIMARY KEY (entry_id, context)
);

-- Indexes
CREATE INDEX idx_entries_tier ON entries(tier);
CREATE INDEX idx_entries_type ON entries(type);
CREATE INDEX idx_entries_boost ON entries(boost_weight DESC);
CREATE INDEX idx_aliases_alias ON aliases(alias);
CREATE INDEX idx_contexts_context ON entry_contexts(context);
```

### 2.3 Tier System

| Tier | Description | Boost | Load Policy |
|------|-------------|-------|-------------|
| **A** | Always-on core (you, family, daily tech) | 3.0 | Every session |
| **B** | Running (races, metrics, coaches) | 2.5 | Context: running |
| **C** | Work @ Veeva (products, pharma jargon) | 2.5 | Context: work |
| **D** | ASR/AI tech (models, libraries) | 2.0 | Context: asr_dev |
| **E** | Health/wearables (devices, metrics) | 2.0 | Context: health |
| **F** | Investing (tickers, macro) | 1.5 | Context: investing |
| **G** | Travel/lifestyle (places, restaurants) | 1.5 | Context: travel |
| **H** | Gear (cameras, watches) | 1.0 | Context: gear |

---

## 3. Implementation Phases

### Phase 1: Core Infrastructure (Foundation)
**Goal:** Replace flat files with SQLite + rich entry model

#### 1.1 Database Layer
- [ ] Create `src/asr/dictionary/models.py` - Pydantic models for Entry, Alias, Pronunciation
- [ ] Create `src/asr/dictionary/db.py` - SQLite operations (CRUD, search, bulk import)
- [ ] Create `src/asr/dictionary/migrations.py` - Schema creation and upgrades
- [ ] Add migration from existing `~/.asr/vocabularies/*.txt` to new DB

#### 1.2 Entry Management
- [ ] Create `src/asr/dictionary/manager.py` - High-level dictionary operations
- [ ] Implement entry CRUD with alias/pronunciation/context handling
- [ ] Implement fuzzy search (for correction candidate lookup)
- [ ] Add bulk import from JSON/CSV with deduplication

#### 1.3 CLI Commands
- [ ] `asr dict init` - Initialize database, run migrations
- [ ] `asr dict add` - Add entry with full metadata
- [ ] `asr dict search <query>` - Fuzzy search entries
- [ ] `asr dict import <file>` - Bulk import from JSON
- [ ] `asr dict export` - Export to JSON for backup/sharing
- [ ] `asr dict stats` - Show tier counts, coverage

### Phase 2: Seed Data (Mike-World Vocabularies)
**Goal:** Populate with tiered proper noun data

#### 2.1 Tier A - Always-On Core
```python
TIER_A_ENTRIES = [
    # Personal
    {"canonical": "Mike", "type": "person", "tier": "A", "boost": 3.0},
    {"canonical": "Mike Edwards", "type": "person", "tier": "A", "boost": 3.0},
    {"canonical": "Superfit", "type": "product", "tier": "A", "boost": 3.0},
    {"canonical": "Project Sub-3", "type": "jargon", "tier": "A", "boost": 3.0,
     "aliases": ["sub-3", "sub 3", "Sub-Three"]},
    {"canonical": "Finney", "type": "misc", "tier": "A", "boost": 3.0,
     "aliases": ["Finney Potter"]},

    # Daily tech stack
    {"canonical": "Apple Watch Ultra", "type": "product", "tier": "A", "boost": 3.0,
     "aliases": ["Watch Ultra", "Ultra"]},
    {"canonical": "MacBook Air M4", "type": "product", "tier": "A", "boost": 3.0,
     "aliases": ["MacBook Air", "M4 Air"]},
    {"canonical": "Oura Ring 4", "type": "product", "tier": "A", "boost": 3.0,
     "aliases": ["Oura Ring", "Oura"]},
    {"canonical": "Garmin", "type": "org", "tier": "A", "boost": 3.0},
    {"canonical": "Whoop", "type": "product", "tier": "A", "boost": 3.0,
     "aliases": ["Whoop 5.0", "Whoop MG"]},

    # AI tools
    {"canonical": "ChatGPT", "type": "product", "tier": "A", "boost": 3.0},
    {"canonical": "Claude", "type": "product", "tier": "A", "boost": 3.0,
     "aliases": ["Claude Sonnet", "Claude Opus"]},
    {"canonical": "Superwhisper", "type": "product", "tier": "A", "boost": 3.0},
    {"canonical": "Granola", "type": "product", "tier": "A", "boost": 3.0},
    {"canonical": "NotebookLM", "type": "product", "tier": "A", "boost": 3.0},
    {"canonical": "Polymarket", "type": "product", "tier": "A", "boost": 3.0},
]
```

#### 2.2 Domain Pack Files
- [ ] Create `seeds/tier_a_core.json` - Always-on (50-100 entries)
- [ ] Create `seeds/tier_b_running.json` - Running domain (~80 entries)
- [ ] Create `seeds/tier_c_veeva.json` - Work/pharma (~60 entries)
- [ ] Create `seeds/tier_d_asr_tech.json` - ASR/AI tech (~50 entries)
- [ ] Create `seeds/tier_e_health.json` - Health/wearables (~40 entries)
- [ ] Create `seeds/tier_f_investing.json` - Investing (~30 entries)
- [ ] Create `seeds/tier_g_travel.json` - Travel/lifestyle (~40 entries)
- [ ] Create `seeds/tier_h_gear.json` - Cameras/watches (~20 entries)

### Phase 3: Context-Aware Selection
**Goal:** Generate ranked, pruned bias lists per session

#### 3.1 Context Profiles
```json
// ~/.asr/dictionaries/contexts/running.json
{
    "name": "running",
    "description": "Running, training, races",
    "include_tiers": ["A", "B", "E"],  // Core + Running + Health
    "include_contexts": ["running", "health"],
    "max_entries": 150,
    "boost_multiplier": 1.2
}
```

#### 3.2 Selector Module
- [ ] Create `src/asr/dictionary/selector.py`
- [ ] Implement `select_bias_list(context: str, max_entries: int) -> list[Entry]`
- [ ] Scoring: `score = boost_weight * tier_weight * recency_decay * context_match`
- [ ] Rank by score, return top-k (default 150, max 500)
- [ ] Generate both canonical + key aliases (not all)

#### 3.3 CLI Integration
- [ ] `asr dict context list` - Show available contexts
- [ ] `asr dict context show <name>` - Show context config
- [ ] `asr dict context create <name>` - Interactive context creation
- [ ] `asr transcribe --context running` - Use context for bias list

### Phase 4: Whisper Integration
**Goal:** Generate optimized initial_prompt from dictionary

#### 4.1 Prompt Generator
- [ ] Create `src/asr/dictionary/prompt_generator.py`
- [ ] `generate_whisper_prompt(entries: list[Entry]) -> str`
- [ ] Format: `"In this conversation, you may hear: Navesink Challenge, Superfit, Veeva Vault, Garmin Forerunner 970, ..."`
- [ ] Truncate to ~200 tokens (Whisper prompt limit)
- [ ] Prioritize by boost weight, include key aliases

#### 4.2 Config Integration
- [ ] Add `dictionary_context` to `ASRConfig`
- [ ] Auto-load context on transcribe if specified
- [ ] Merge with existing `--prompt` flag (user prompt takes priority)

### Phase 5: Claude Correction Integration
**Goal:** Pass dictionary entries to correction pipeline

#### 5.1 Correction Prompt Enhancement
- [ ] Modify `nlp/prompts.py` to accept dictionary entries
- [ ] Add dictionary section to Pass 1 prompt:
  ```
  ## Known Proper Nouns (use these exact spellings)
  - Navesink Challenge (aliases: Navesink, Navesink 12K)
  - Ron Chernow (type: person, historian)
  - ...
  ```
- [ ] Limit to top 50-100 entries in correction prompt (cost control)

#### 5.2 Candidate Matching
- [ ] Create `src/asr/dictionary/matcher.py`
- [ ] `find_candidates(text: str, entries: list[Entry]) -> list[Match]`
- [ ] Use edit distance + phonetic similarity (Soundex/Metaphone)
- [ ] Return ranked candidates for each suspected proper noun span

#### 5.3 Post-Correction Feedback
- [ ] When Claude corrects a term to a dictionary entry, boost occurrence_count
- [ ] When Claude corrects to unknown term, add to pending queue
- [ ] Track which entries are most useful (for pruning cold entries)

### Phase 6: LLM Post-Correction Pass (Optional)
**Goal:** Dedicated proper noun correction with full dictionary access

#### 6.1 Dictionary-Aware Correction
- [ ] Create `src/asr/nlp/noun_corrector.py`
- [ ] Pass 3 (optional): Focus specifically on proper nouns
- [ ] Input: transcript + full dictionary excerpt for suspected nouns
- [ ] Output: corrected proper nouns with confidence + reasoning

#### 6.2 Entity Disambiguation
- [ ] When multiple candidates match, ask Claude to disambiguate
- [ ] Use context (surrounding text, speaker, topic) to choose
- [ ] Example: "Finley" → coach or bird? Context: running training → coach

### Phase 7: Feedback Loop & Learning
**Goal:** Continuously improve dictionary from corrections

#### 7.1 Occurrence Tracking
- [ ] Update `last_seen_at` and `occurrence_count` when entry is used in correction
- [ ] Decay unused entries (reduce boost over time)
- [ ] Auto-archive entries with 0 occurrences in 6 months

#### 7.2 Alias Learning
- [ ] When ASR consistently produces variant X for entry Y, add X as alias
- [ ] Example: ASR outputs "Ron Charno" for "Ron Chernow" → add as misspelling alias
- [ ] Requires 3+ occurrences to auto-add

#### 7.3 New Entry Discovery
- [ ] Extract proper nouns from corrected transcripts via NER
- [ ] If not in dictionary and appears 3+ times, suggest as new entry
- [ ] Manual approval via `asr dict pending`

---

## 4. File Structure (Final)

```
src/asr/
├── dictionary/
│   ├── __init__.py
│   ├── models.py         # Pydantic: Entry, Alias, Pronunciation, Context
│   ├── db.py             # SQLite operations
│   ├── migrations.py     # Schema management
│   ├── manager.py        # High-level CRUD
│   ├── selector.py       # Context-aware selection
│   ├── prompt_generator.py  # Whisper prompt formatting
│   └── matcher.py        # Fuzzy/phonetic candidate matching
├── vocabularies/         # (Legacy - keep for backward compat)
│   ├── common_names.py
│   └── tech_terms.py
└── nlp/
    ├── corrector.py      # (Modified) Accept dictionary entries
    ├── prompts.py        # (Modified) Dictionary section in prompts
    └── noun_corrector.py # (New) Dedicated proper noun pass

seeds/
├── tier_a_core.json
├── tier_b_running.json
├── tier_c_veeva.json
├── tier_d_asr_tech.json
├── tier_e_health.json
├── tier_f_investing.json
├── tier_g_travel.json
└── tier_h_gear.json
```

---

## 5. CLI Commands (Final)

```bash
# Dictionary management
asr dict init                      # Initialize database
asr dict add "Ron Chernow" --type person --tier B --context biography
asr dict add "Navesink Challenge" --aliases "Navesink,Navesink 12K" --tier B
asr dict search "charno"           # Fuzzy search
asr dict show "Ron Chernow"        # Show entry details
asr dict remove "old-entry"        # Remove entry
asr dict import seeds/tier_b_running.json
asr dict export --tier A --format json > backup.json
asr dict stats                     # Tier counts, coverage

# Context management
asr dict context list              # List available contexts
asr dict context show running      # Show context config
asr dict context create work       # Create new context

# Transcription with dictionary
asr transcribe file.m4a --context running  # Use running context
asr transcribe file.m4a --context work --correct --domain technical

# Pending entries (learning)
asr dict pending                   # Review pending entries
asr dict approve "New Term"        # Approve pending entry
asr dict reject "Hallucinated"     # Reject pending entry
```

---

## 6. Integration Points

### 6.1 Whisper (Pre-ASR)
```python
# In backends/crisperwhisper_engine.py
def transcribe(self, chunk: AudioChunk, config: ASRConfig) -> TranscriptSegment:
    # Generate bias prompt from dictionary
    if config.dictionary_context:
        entries = selector.select_bias_list(config.dictionary_context, max_entries=150)
        bias_prompt = prompt_generator.generate_whisper_prompt(entries)
        # Merge with user prompt
        full_prompt = f"{bias_prompt}\n{config.prompt or ''}"
    else:
        full_prompt = config.prompt

    result = mlx_whisper.transcribe(
        chunk.path,
        initial_prompt=full_prompt,
        ...
    )
```

### 6.2 Claude Correction (During/Post-ASR)
```python
# In nlp/corrector.py
def correct_segments(segments: list[Segment], config: CorrectionConfig) -> list[Segment]:
    # Get dictionary entries for correction context
    if config.dictionary_context:
        entries = selector.select_bias_list(config.dictionary_context, max_entries=100)
        dictionary_block = format_dictionary_for_prompt(entries)
    else:
        dictionary_block = ""

    # Include in Pass 1 prompt
    prompt = build_pass1_prompt(
        segments=segments,
        domain=config.domain,
        dictionary=dictionary_block,  # NEW
    )
```

### 6.3 Feedback Loop
```python
# In nlp/corrector.py (after successful correction)
def record_dictionary_usage(original: str, corrected: str, entries: list[Entry]):
    for entry in entries:
        if entry.canonical == corrected or corrected in entry.aliases:
            # Boost this entry
            manager.increment_occurrence(entry.id)
            manager.update_last_seen(entry.id)

            # If original was a consistent misspelling, add as alias
            if original != corrected:
                manager.maybe_add_alias(entry.id, original, is_misspelling=True)
```

---

## 7. Performance Considerations

### 7.1 Prompt Size Limits
- Whisper `initial_prompt`: ~200 tokens effective, 224 max
- Claude correction prompt: ~4000 tokens budget for dictionary section
- **Strategy**: Top-k selection, canonical + 1-2 key aliases only

### 7.2 Database Performance
- SQLite with WAL mode for concurrent reads
- Indexes on tier, type, boost_weight, context
- Cache hot entries in memory during session

### 7.3 Selection Performance
- Pre-compute tier scores at load time
- Lazy-load pronunciations (rarely needed)
- Cache context bias lists (invalidate on dictionary update)

---

## 8. Migration Path

### From Current System
1. Run `asr dict init` to create new database
2. Auto-import existing `~/.asr/vocabularies/*.txt` files
3. Existing `--prompt` flag continues to work
4. New `--context` flag is additive
5. Old vocab commands (`asr vocab *`) remain for backward compat

### Gradual Rollout
1. **Week 1**: Phase 1 (infrastructure) + Phase 2 (seed data for Tier A only)
2. **Week 2**: Phase 3 (context selection) + Phase 4 (Whisper integration)
3. **Week 3**: Phase 5 (Claude integration) + remaining seed data
4. **Week 4**: Phase 6-7 (post-correction, feedback loop)

---

## 9. Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Proper noun WER | ~15% | <5% |
| Correction acceptance rate | 70% | 85% |
| Dictionary-assisted corrections | 0% | 60% |
| User corrections needed | ~20/transcript | <5/transcript |

### How to Measure
- Log proper noun recognition accuracy per session
- Track which dictionary entries were used in corrections
- Count user edits to proper nouns in final output

---

## 10. Open Questions / Decisions Needed

1. **Pronunciation data**: Include IPA/phoneme sequences now, or defer?
   - Pro: Enables phonetic matching, future G2P integration
   - Con: More work to populate, may not be needed for Whisper biasing

2. **Context auto-detection**: Auto-select context based on content?
   - Example: Detect "Strava" in audio → enable running context
   - Adds complexity, may have false positives

3. **Multi-user support**: Separate dictionaries per user/profile?
   - Current design assumes single user
   - Could add user_id to entries table

4. **Cloud sync**: Sync dictionary across devices?
   - Could use iCloud/Dropbox for `~/.asr/dictionaries/`
   - Or build explicit export/import workflow

---

## Appendix A: Sample Seed Data (Tier B - Running)

```json
[
  {
    "canonical": "Garmin Forerunner 970",
    "display": "Garmin Forerunner 970",
    "type": "product",
    "tier": "B",
    "boost_weight": 2.5,
    "aliases": ["Forerunner 970", "FR 970", "FR970"],
    "contexts": ["running", "health"]
  },
  {
    "canonical": "Navesink Challenge",
    "display": "Navesink Challenge 12K",
    "type": "event",
    "tier": "B",
    "boost_weight": 2.5,
    "aliases": ["Navesink", "Navesink 12K", "Navesink Challenge 12K"],
    "contexts": ["running"]
  },
  {
    "canonical": "TrainingPeaks",
    "type": "product",
    "tier": "B",
    "boost_weight": 2.0,
    "contexts": ["running"]
  },
  {
    "canonical": "VO2max",
    "display": "VO2max",
    "type": "jargon",
    "tier": "B",
    "boost_weight": 2.5,
    "aliases": ["VO2 max", "V O 2 max"],
    "contexts": ["running", "health"]
  },
  {
    "canonical": "Norwegian double threshold",
    "type": "jargon",
    "tier": "B",
    "boost_weight": 2.0,
    "aliases": ["double threshold", "Norwegian method"],
    "contexts": ["running"]
  }
]
```

## Appendix B: Whisper Prompt Template

```
In this conversation, you may hear the following names and terms:

People: Mike Edwards, Ron Chernow, Finley (coach), Kirby, Holder
Products: Garmin Forerunner 970, Oura Ring 4, Whoop 5.0, MacBook Air M4
Companies: Veeva, Anthropic, OpenAI, Garmin
Events: Navesink Challenge, Hamburg Marathon
Places: Hartshorne Woods, Amelia Island, Fort Clinch
Terms: VO2max, body battery, decoupling, sub-3, cGMP, CAPA

Please transcribe accurately, preserving these exact spellings.
```

## Appendix C: Claude Correction Dictionary Block

```markdown
## Known Proper Nouns (use these exact spellings when correcting)

| Term | Type | Aliases | Notes |
|------|------|---------|-------|
| Ron Chernow | person | | Historian, biographer |
| Navesink Challenge | event | Navesink, Navesink 12K | Running race |
| VO2max | jargon | VO2 max | Fitness metric |
| Veeva Vault | product | Vault | Enterprise software |
| MacBook Air M4 | product | MacBook Air, M4 Air | Apple laptop |

When you see text that sounds like one of these terms, use the canonical spelling.
Only substitute if you're confident the audio matches - do not force-fit.
```
