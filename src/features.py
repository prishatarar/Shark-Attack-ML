"""Feature engineering helpers."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .mappings import REEF_SHARKS


def replace_activity(activity: object) -> str:
    """Bucket free-form activities into a smaller set of categories."""
    activity_lower = str(activity).lower()
    if "surf" in activity_lower:
        return "surfing"
    if "fish" in activity_lower:
        return "fishing"
    if "board" in activity_lower:
        return "boarding"
    if "swim" in activity_lower:
        return "swimming"
    if "div" in activity_lower:
        return "diving"
    return "other"


def classify_shark(species: object) -> str:
    """Collapse species strings into a manageable set of labels."""
    species_lower = str(species).lower()
    if "white" in species_lower and "shark" in species_lower:
        return "White Shark"
    if "tiger" in species_lower and "shark" in species_lower:
        return "Tiger Shark"
    if "bull" in species_lower and "shark" in species_lower:
        return "Bull Shark"
    if "blue" in species_lower and "shark" in species_lower:
        return "Blue Shark"
    if "tip" in species_lower and "shark" in species_lower:
        return "Whitetip Shark"
    if any(shark.lower() in species_lower for shark in REEF_SHARKS) or "reef" in species_lower:
        return "Reef Shark"
    if "nurse" in species_lower and "shark" in species_lower:
        return "Nurse Shark"
    if ("no" in species_lower or "not" in species_lower) and "involvement" in species_lower:
        return "No Shark"
    return "Other Species"


def extract_date(case_number: object) -> Optional[str]:
    """Extract a normalized YYYY.MM.DD date from the notebook's case-number field."""
    case_str = str(case_number).strip()
    match = re.match(r"(\d{4})[.-](\d{2})[.-](\d{2})", case_str)
    if not match:
        return None
    year, month, day = match.groups()
    month = "01" if month == "00" else month
    day = "01" if day == "00" else day
    return f"{year}.{month}.{day}"


def categorize_time(value: object) -> str | float:
    """Map raw time strings into broad time-of-day buckets."""
    value = str(value).lower().strip()

    if "morning" in value:
        return "morning"
    if "afternoon" in value:
        return "afternoon"
    if "evening" in value:
        return "evening"
    if "night" in value:
        return "night"

    hhmm_match = re.match(r"^(\d{1,2})h(\d{2})$", value)
    if hhmm_match:
        hour = int(hhmm_match.group(1))
        minute = int(hhmm_match.group(2))
        total_minutes = hour * 60 + minute
        if 300 <= total_minutes < 720:
            return "morning"
        if 720 <= total_minutes < 1020:
            return "afternoon"
        if 1020 <= total_minutes < 1260:
            return "evening"
        return "night"

    hour_minute_match = re.search(r"(\d{1,2})\s*hour[s]?\s*(\d{1,2})?\s*minute[s]?", value)
    if hour_minute_match:
        hour = int(hour_minute_match.group(1))
        minute = int(hour_minute_match.group(2)) if hour_minute_match.group(2) else 0
        total_minutes = hour * 60 + minute
        if 300 <= total_minutes < 720:
            return "morning"
        if 720 <= total_minutes < 1020:
            return "afternoon"
        if 1020 <= total_minutes < 1260:
            return "evening"
        return "night"

    return np.nan


def get_season(date: pd.Timestamp) -> str | float:
    """Return meteorological season for a pandas timestamp."""
    if pd.isna(date):
        return np.nan
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"
