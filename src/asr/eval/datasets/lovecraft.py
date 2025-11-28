"""LibriVox Lovecraft Dataset for ASR Evaluation.

H.P. Lovecraft's works are public domain (died 1937).
LibriVox recordings are public domain (CC0).
Reference texts from Wikisource are public domain.

Audio: ~20-30 min per story, clear narration, period vocabulary
Difficulty: Medium-Hard (archaic language, proper nouns, cosmic horror terms)
"""

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LovecraftStory:
    """Metadata for a Lovecraft story in the evaluation set."""

    name: str
    slug: str  # URL-safe identifier
    duration_minutes: float
    audio_url: str  # Archive.org direct link
    text_code: str  # hplovecraft.com text code (e.g., "d" for Dagon)
    difficulty: str  # easy, medium, hard


# Evaluation dataset: 5 stories, ~2 hours total
# Selected for variety in length, vocabulary, and recording quality
# Audio URLs verified 2024 from Archive.org and Wikimedia Commons
# Text from hplovecraft.com (public domain)
LOVECRAFT_STORIES = [
    LovecraftStory(
        name="Dagon",
        slug="dagon",
        duration_minutes=14.9,
        audio_url="https://archive.org/download/collected_lovecraft_0810_librivox/dagon_lovecraft_mras.mp3",
        text_code="d",  # https://www.hplovecraft.com/writings/texts/fiction/d.aspx
        difficulty="easy",  # Shorter, clear narration
    ),
    LovecraftStory(
        name="The Outsider",
        slug="the_outsider",
        duration_minutes=16.5,
        audio_url="https://archive.org/download/TheOutsiderByHPLovecraft/TheOutsiderByHPLovecraft.mp3",
        text_code="o",  # https://www.hplovecraft.com/writings/texts/fiction/o.aspx
        difficulty="medium",
    ),
    LovecraftStory(
        name="Cool Air",
        slug="cool_air",
        duration_minutes=23.0,
        audio_url="https://archive.org/download/ghohor075_2411_librivox/ghohor075_coolair_lovecraft_dw_128kb.mp3",
        text_code="ca",  # https://www.hplovecraft.com/writings/texts/fiction/ca.aspx
        difficulty="medium",
    ),
    LovecraftStory(
        name="The Moon-Bog",
        slug="the_moon_bog",
        duration_minutes=22.0,
        audio_url="https://archive.org/download/AudioReadingOftheMoon-bogByH.p.Lovecraft/hpl-moon-bog.mp3",
        text_code="mb",  # https://www.hplovecraft.com/writings/texts/fiction/mb.aspx
        difficulty="hard",  # Irish names, archaic terms
    ),
    LovecraftStory(
        name="He",
        slug="he",
        duration_minutes=29.75,
        audio_url="https://upload.wikimedia.org/wikipedia/commons/7/73/LibriVox_-_He_%28Lovecraft%29.mp3",
        text_code="he",  # https://www.hplovecraft.com/writings/texts/fiction/he.aspx
        difficulty="medium",
    ),
]


def download_lovecraft_dataset(
    output_dir: Path,
    stories: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download LibriVox Lovecraft audio files.

    Args:
        output_dir: Directory to save audio files
        stories: List of story slugs to download (None = all)
        force: Re-download even if files exist

    Returns:
        List of paths to downloaded audio files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for story in LOVECRAFT_STORIES:
        if stories and story.slug not in stories:
            continue

        # Determine file extension from URL
        ext = ".ogg" if story.audio_url.endswith(".ogg") else ".mp3"
        output_path = output_dir / f"{story.slug}{ext}"

        if output_path.exists() and not force:
            print(f"Skipping {story.name} (already exists)")
            downloaded.append(output_path)
            continue

        print(f"Downloading {story.name}...")
        try:
            # Wikimedia requires a User-Agent header
            req = urllib.request.Request(
                story.audio_url,
                headers={"User-Agent": "ASR-Eval/1.0 (audio transcription tool)"}
            )
            with urllib.request.urlopen(req) as response:
                with open(output_path, "wb") as out_file:
                    out_file.write(response.read())
            downloaded.append(output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error downloading {story.name}: {e}")

    return downloaded


# Reference text URLs (canonical sources)
# hplovecraft.com for stories not on Project Gutenberg
# Project Gutenberg plain text for those available
REFERENCE_TEXT_URLS = {
    "dagon": "https://www.hplovecraft.com/writings/texts/fiction/d.aspx",
    "the_outsider": "https://www.hplovecraft.com/writings/texts/fiction/o.aspx",
    "cool_air": "https://www.gutenberg.org/ebooks/73177.txt.utf-8",
    "the_moon_bog": "https://www.hplovecraft.com/writings/texts/fiction/mb.aspx",
    "he": "https://www.gutenberg.org/ebooks/68547.txt.utf-8",
}


def get_lovecraft_reference(story_slug: str, cache_dir: Path | None = None) -> str:
    """Get reference text for a Lovecraft story.

    Fetches from hplovecraft.com or Project Gutenberg depending on availability.

    Args:
        story_slug: Story identifier (e.g., "the_outsider")
        cache_dir: Optional directory to cache downloaded texts

    Returns:
        Cleaned reference text
    """
    # Find story metadata
    story = None
    for s in LOVECRAFT_STORIES:
        if s.slug == story_slug:
            story = s
            break

    if not story:
        raise ValueError(f"Unknown story: {story_slug}")

    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"{story_slug}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

    # Get URL for this story
    text_url = REFERENCE_TEXT_URLS.get(story_slug)
    if not text_url:
        print(f"No reference text URL for {story.name}")
        return ""

    print(f"Fetching reference text for {story.name}...")

    try:
        req = urllib.request.Request(
            text_url,
            headers={"User-Agent": "ASR-Eval/1.0 (audio transcription tool)"}
        )
        with urllib.request.urlopen(req) as response:
            raw_text = response.read().decode("utf-8")

        # Clean based on source
        if "gutenberg.org" in text_url:
            text = _clean_gutenberg_text(raw_text)
        else:
            text = _clean_hplovecraft_text(raw_text)

        # Cache if directory provided
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{story_slug}.txt"
            cache_path.write_text(text, encoding="utf-8")

        return text

    except Exception as e:
        print(f"Error fetching reference text: {e}")
        return ""


def _clean_gutenberg_text(text: str) -> str:
    """Clean Project Gutenberg plain text.

    Removes:
    - Header/footer boilerplate
    - Extra whitespace
    """
    if not text:
        return ""

    # Find start of actual text (after Gutenberg header)
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the end of this line
            newline_idx = text.find("\n", idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    text = text[start_idx:end_idx]

    # Clean up whitespace
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    # Join with single spaces (prose)
    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _clean_hplovecraft_text(html: str) -> str:
    """Clean hplovecraft.com HTML page to extract story text.

    Removes:
    - HTML tags
    - Navigation/header content
    - Extra whitespace
    """
    if not html:
        return ""

    # Extract text between story markers
    # The stories are typically in <div class="story"> or just the main content
    # Try to find the story text - it usually starts after the title

    # Remove script and style tags
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Replace <br> and <p> with newlines
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</p>", "\n", html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    html = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    html = html.replace("&nbsp;", " ")
    html = html.replace("&amp;", "&")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&quot;", '"')
    html = html.replace("&#39;", "'")
    html = html.replace("&mdash;", "—")
    html = html.replace("&ndash;", "–")

    # Clean up lines
    lines = html.split("\n")
    cleaned_lines = []
    in_story = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip navigation/header content
        skip_patterns = [
            "home", "fiction", "poetry", "essays", "letters",
            "miscellany", "about", "copyright", "h.p. lovecraft",
            "return to", "written", "first published",
        ]
        if any(p in line.lower() for p in skip_patterns) and len(line) < 100:
            continue

        # Story typically starts with first substantial paragraph
        if len(line) > 50:
            in_story = True

        if in_story:
            cleaned_lines.append(line)

    # Join with spaces
    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Vocabulary hints for Lovecraft stories (proper nouns and archaic terms)
LOVECRAFT_VOCABULARY = [
    # Character/Place names
    "Arkham", "Innsmouth", "Miskatonic", "Cthulhu", "Nyarlathotep",
    "Azathoth", "Yog-Sothoth", "Dagon", "R'lyeh", "Yuggoth",
    "Dunwich", "Kingsport", "Providence", "Necronomicon",
    "Abdul Alhazred", "Randolph Carter", "Herbert West",

    # Archaic/unusual terms
    "eldritch", "cyclopean", "gibbous", "gambrel", "blasphemous",
    "antediluvian", "squamous", "rugose", "foetid", "charnel",
    "daemoniac", "noisome", "miasma", "effluvium", "tenebrous",
    "ichor", "batrachian", "abysmal", "furtive", "loathsome",

    # Period terms
    "gasoline", "motor-car", "aeroplane", "gramophone",
    "phosphorescence", "luminescence",
]
