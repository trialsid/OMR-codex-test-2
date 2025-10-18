"""Utility for loading OMR sheet configuration from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from omr_config import SheetLayout


def load_sheet_config(config_path: Path | str) -> SheetLayout:
    """Load sheet configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        SheetLayout instance with values from the config file

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file contains invalid values
        json.JSONDecodeError: If config file is not valid JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # Create SheetLayout with values from config
    # Only pass parameters that are actually in the JSON to allow partial configs
    valid_params = {}

    if "class_options" in config_dict:
        valid_params["class_options"] = config_dict["class_options"]
    if "class_section_options" in config_dict:
        valid_params["class_section_options"] = config_dict["class_section_options"]
    if "roll_columns" in config_dict:
        valid_params["roll_columns"] = config_dict["roll_columns"]
    # Note: roll_rows removed - always 10 for digits 0-9
    if "set_options" in config_dict:
        valid_params["set_options"] = config_dict["set_options"]
    if "question_options" in config_dict:
        valid_params["question_options"] = config_dict["question_options"]
    if "max_questions" in config_dict:
        valid_params["max_questions"] = config_dict["max_questions"]

    try:
        return SheetLayout(**valid_params)
    except (TypeError, AssertionError) as e:
        raise ValueError(f"Invalid configuration values: {e}")


def save_sheet_config(sheet: SheetLayout, config_path: Path | str) -> None:
    """Save sheet configuration to a JSON file.

    Args:
        sheet: SheetLayout instance to save
        config_path: Path where JSON file will be saved
    """
    config_path = Path(config_path)

    config_dict = {
        "class_options": sheet.class_options,
        "class_section_options": sheet.class_section_options,
        "roll_columns": sheet.roll_columns,
        "set_options": sheet.set_options,
        "question_options": sheet.question_options,
    }

    # Only include max_questions if it's set
    if sheet.max_questions is not None:
        config_dict["max_questions"] = sheet.max_questions

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
