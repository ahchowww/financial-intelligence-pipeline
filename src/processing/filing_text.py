from pathlib import Path

from bs4 import BeautifulSoup

import re

def extract_filing_text(
    html_path: Path,
) -> str:
    """
    Convert one SEC filing HTML document into readable plain text.
    """

    # Read the entire .htm file
    html = html_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    # Parse HTML
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove content that is not useful for natural-language analysis
    for tag in soup(
        [
            "script",
            "style",
        ]
    ):
        tag.decompose()

    # Convert HTML to plain text
    text = soup.get_text(
        separator="\n"
    )

    # Remove excessive whitespace
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    # Remove empty lines
    lines = [
        line
        for line in lines
        if line
    ]

    # Join cleaned lines
    clean_text = "\n".join(
        lines
    )

    return clean_text


def extract_10q_mda(
        text: str,
) -> str:
    """
    Extract the actual MD&A section from a 10-Q.

    SEC filings often contain the Item 2 / Item 3 headings twice:
        1. Table of contents
        2. Actual filing content
    
    Strategy:
        1. Find every Item 2 heading.
        2. Keep Item 2 headings whose nearby text
        contains Management's Discussion and Analysis.
        3. Find Item 3 headings referring to
        Quantitative and Qualitative Disclosures.
        4. Build Item 2 -> next Item 3 candidates.
        5. Ignore short table-of-contents candidates.
        6. Return the longest substantial candidate.
    """

    # 1. Find all Item 2 and Item 3 occurrences
    item2_matches = list(
        re.finditer(
            r"\bi\s*tem\s*2\s*\.?",
            text,
            flags=re.IGNORECASE,
        )
    )

    item3_matches = list(
        re.finditer(
            r"\bi\s*tem\s*3\s*\.?",
            text,
            flags=re.IGNORECASE,
        )
    )

    
    # 2. Identify Item 2 headings that are MD&A
    mda_starts = []

    for match in item2_matches:
        position = match.start()

        nearby = (
            text[position:position + 300]
            .lower()
            .replace(
                "’",
                "'",
            )
        )

        # remove all whitespace
        # e.g. Management's Discussion and Analysis of Financial Condition
        # -> management'sdiscussionandanalysisoffinancialcondition
        compact_nearby = re.sub(
            r"\s+",
            "",
            nearby,
        )

        if (
            (
                "management'sdiscussionandanalysis" in compact_nearby
            )
            and
            (
                "financialcondition"in compact_nearby
            )
        ):
            mda_starts.append(position)

    if not mda_starts:
        raise ValueError(
            "Could not find any Item 2 MD&A headings."
        )

    
    # 3. Identify Item 3 market-risk headings
    item3_ends = []

    for match in item3_matches:
        position = match.start()

        nearby = (
            text[position:position + 300]
            .lower()
        )

        compact_nearby = re.sub(
            r"\s+",
            "",
            nearby,
        )

        if (
            "quantitative" in compact_nearby
            and
            "qualitative" in compact_nearby
            and
            "marketrisk" in compact_nearby
        ):
            item3_ends.append(position)

    if not item3_ends:
        raise ValueError(
            "Could not find any Item 3 market-risk headings."
        )

    
    # 4. Build valid Item 2 -> Item 3 candidates
    candidates = []

    for start in mda_starts:
        # Find every Item 3 that comes after this Item 2
        possible_ends = [
            end
            for end in item3_ends
            if end > start
        ]

        if not possible_ends:
            continue

        # choose nearest Item 3
        end = min(possible_ends)

        # Extract section
        section = (
            text[start:end]
            .strip()
        )

        # Save candidate metadata
        candidates.append(
            {
                "start": start,
                "end": end,
                "length": len(section),
                "text": section,
            }
        )

    if not candidates:
        raise ValueError(
            "Could not build any MD&A section candidates."
        )

    
    # 5. Remove TOC candidates
    substantial_candidates = [
        candidate
        for candidate in candidates
        if candidate["length"] >= 1000
    ]

    if not substantial_candidates:
        details = [
            (
                candidate["start"],
                candidate["end"],
                candidate["length"],
            )
            for candidate in candidates
        ]

        raise ValueError(
            "MD&A candidates were found, but none were substantial enough. "
            f"Candidates: {details}"
        )

    
    # 6. Select actual MD&A
    best_candidate = max(
        substantial_candidates,
        key=lambda candidate:
            candidate["length"],
    )

    return best_candidate["text"]


def extract_10k_mda(
        text: str,
) -> str:
    """
    Extract the actual MD&A section from a 10-K.

    Strategy:
    1. Find every Item 7 occurrence.
    2. Keep Item 7 occurrences whose nearby test contains
    Management's Discussion and Analysis.
    3. Find Item 7A headings referring to 
    Quantitative and Qualitative Disclosures.
    4. Build Item 7 -> next Item 7A candidates.
    5. Reject short table-of-contents candidates.
    6. Return the longest substantial candidate.
    """

    # 1. Find every Item 7 and Item 7A occurrence
    item7_matches = list(
        re.finditer(
            r"\bi\s*tem\s*7(?!\s*a)\s*\.?",
            text,
            flags=re.IGNORECASE,
        )
    )

    item7a_matches = list(
            re.finditer(
                r"\bi\s*tem\s*7\s*a\s*\.?",
                text,
                flags=re.IGNORECASE,
            )
        )

    # 2. Identify Item 7 MD&A headings
    mda_starts = []

    for match in item7_matches:
        position = match.start()

        nearby = (
            text[position: position + 300]
            .lower()
            .replace(
                "’",
                "'",
            )
        )

        compact_nearby = re.sub(
            r"\s+",
            "",
            nearby,
        )

        if (
            "management'sdiscussionandanalysis" in compact_nearby
            and
            "financialcondition" in compact_nearby
        ):
            mda_starts.append(position)

    if not mda_starts:
        raise ValueError(
            "Could not find any Item 7 MD&A headings."
        )

    # 3. Identify Item 7A market-risk headings
    item7a_ends = []

    for match in item7a_matches:
        position = match.start()

        nearby = (
            text[position: position + 300]
            .lower()
        )

        compact_nearby = re.sub(
            r"\s+",
            "",
            nearby,
        )

        if (
            "quantitative" in compact_nearby
            and
            "qualitative" in compact_nearby
            and
            "marketrisk" in compact_nearby
        ):
            item7a_ends.append(position)

    if not item7a_ends:
        raise ValueError(
            "Could not find any Item 7A market-risk headings."
        )

    # 4. Build Item 7 -> Item 7A candidates
    candidates = []

    for start in mda_starts:
        possible_ends = [
            end
            for end in item7a_ends
            if end > start
        ]

        if not possible_ends:
            continue

        end = min(possible_ends)

        section = text[start: end].strip()

        candidates.append(
            {
                "start": start,
                "end": end,
                "length": len(section),
                "text" : section,
            }
        )

    if not candidates:
        raise ValueError(
            "Could not build any 10-K MD&A section candidates."
        )

    # 5. Reject TOC candidates
    substantial_candidates = [
        candidate
        for candidate in candidates
        if candidate["length"] >= 1000
    ]

    if not substantial_candidates:
        details = [
            {
                candidate["start"],
                candidate["end"],
                candidate["length"],
            }
            for candidate in candidates
        ]

        raise ValueError(
            "10-K MD&A candidates were found, but none were substantial enough."
            f"Candidates: {details}"
        )

    # 6. Select the longest substantial candidate
    best_candidate = max(
        substantial_candidates,
        key=lambda candidate:
            candidate["length"],
    )

    return best_candidate["text"]


def extract_mda(
        text: str,
        form: str,
) -> str:
    """
    Extract MD&A from an SEC filing based on form type.

    10-Q:
        Item 2 -> Item 3
    
    10-K:
        Item 7 -> Item 7A
    """

    normalized_form = (
        form
        .strip()
        .upper()
    )

    if normalized_form == "10-Q":
        return extract_10q_mda(text)

    if normalized_form == "10-K":
        return extract_10k_mda(text)

    raise ValueError(
        f"Unsupported filing form: {form}"
    )


def clean_mda_dataset(
        text: str,
) -> str:
    """
    Clean an extracted MD&A section while preserving its 
    actual financial metadata.
    
    Removes: 
    - non-breaking spaces
    - standalone table of contents
    - page numbers directly adjacent to TOC
    - excessive white space
    """

    # 1. Normalize non-breaking spaces
    text = text.replace(
        "\xa0",
        " ",
    )

    # 2. Strip lines and remove empty lines
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    cleaned_lines = []

    # 3. Remove SEC navigation artifacts safely
    for index, line in enumerate(lines):
        lower_line = line.lower()

        # remove standalone navigation text
        if (lower_line == "table of contents"):
            continue

        # only treat a number as a page number
        # when directly adjacent to TOC
        if re.fullmatch(
            r"\d{1,3}",
            line,
        ):
            previous_is_toc = (
                index > 0
                and
                lines[index-1].lower() == "table of contents"
            )

            next_is_toc = (
                index+1 < len(lines)
                and 
                lines[index+1].lower() == "table of contents"
            )

            if (previous_is_toc or next_is_toc):
                continue

        cleaned_lines.append(line)
            
    # 4. Join cleaned lines
    cleaned_text = "\n".join(
        cleaned_lines
    )

    return cleaned_text