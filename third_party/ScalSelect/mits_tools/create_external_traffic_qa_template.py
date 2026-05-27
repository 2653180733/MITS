"""Create an Excel template for external traffic QA annotation with dropdown menus for target and notes."""

from __future__ import annotations

import argparse
import os
from typing import List


DEFAULT_OUTPUT = os.path.join("templates", "external_traffic_qa_template.xlsx")


HEADERS = [
    "image",
    "scene",
    "task",
    "question",
    "answer",
    "target",
    "notes",
]


# --- Example rows updated with standardized target names and meaningful notes ---
EXAMPLE_ROWS = [
    [
        "test_images/0001.jpg",
        "accident",
        "recognition",
        "Is there a traffic accident in the image?",
        "Yes",
        "traffic_accident",
        "annotator:A;confidence:high",
    ],
    [
        "test_images/0001.jpg",
        "accident",
        "counting",
        "How many damaged vehicles are visible?",
        "2",
        "damaged_vehicle",
        "annotator:A;confidence:high",
    ],
    [
        "test_images/0001.jpg",
        "accident",
        "localization",
        "Where is the damaged vehicle? Answer with bbox [x1,y1,x2,y2].",
        "[338,70,664,371]",
        "damaged_vehicle",
        "annotator:A;confidence:medium",
    ],
    [
        "test_images/0002.jpg",
        "construction",
        "recognition",
        "Is there road construction in the image?",
        "Yes",
        "construction_vehicle",
        "annotator:A;confidence:high",
    ],
    [
        "test_images/0002.jpg",
        "construction",
        "counting",
        "How many traffic cones are visible?",
        "5",
        "traffic_cone",
        "annotator:A;confidence:high",
    ],
    [
        "test_images/0002.jpg",
        "construction",
        "localization",
        "Where is the excavator? Answer with bbox [x1,y1,x2,y2].",
        "[200,150,500,400]",
        "excavator",
        "annotator:A;confidence:medium;partial_occlusion:true",
    ],
    [
        "test_images/0003.jpg",
        "firesmoke",
        "recognition",
        "Is there fire or smoke in the image?",
        "Yes",
        "fire_source",
        "annotator:A;confidence:high",
    ],
    [
        "test_images/0003.jpg",
        "firesmoke",
        "counting",
        "How many vehicles are visible?",
        "3",
        "vehicle",
        "annotator:A;confidence:medium",
    ],
    [
        "test_images/0004.jpg",
        "weather",
        "recognition",
        "Is there severe weather in the image?",
        "Yes",
        "snow_covered_road",
        "annotator:A;confidence:high;night_scene:true",
    ],
]


SCENES = [
    "accident",
    "construction",
    "firesmoke",
    "jam",
    "person_vehicle",
    "spill",
    "weather",
    "normal",
    "other",
]

TASKS = [
    "recognition",
    "counting",
    "localization",
    "background",
    "reasoning",
]

# --- Standardized target names, grouped by scene for reference ---
TARGETS = [
    # accident
    "traffic_accident",
    "damaged_vehicle",
    "overturned_vehicle",
    "debris",
    # construction
    "traffic_cone",
    "barricade",
    "construction_vehicle",
    "excavator",
    # person_vehicle / jam
    "car",
    "pedestrian",
    "motorcycle",
    "bus",
    "truck",
    "vehicle",
    # firesmoke
    "fire_source",
    "smoke_plume",
    # spill
    "spill_area",
    # weather
    "snow_covered_road",
    "flooded_road",
    # generic
    "traffic_light",
    "road_sign",
    "guardrail",
]

# --- Common notes combinations ---
NOTES = [
    "",
    "annotator:A;confidence:high",
    "annotator:A;confidence:medium",
    "annotator:A;confidence:low",
    "annotator:B;confidence:high",
    "annotator:B;confidence:medium",
    "annotator:B;confidence:low",
    "annotator:A;confidence:high;night_scene:true",
    "annotator:A;confidence:medium;partial_occlusion:true",
    "annotator:A;confidence:medium;crowded:true",
    "annotator:A;confidence:low;partial_occlusion:true",
    "annotator:A;confidence:low;night_scene:true",
    "annotator:A;confidence:low;crowded:true",
    "annotator:B;confidence:high;night_scene:true",
    "annotator:B;confidence:medium;partial_occlusion:true",
]


GUIDE_ROWS = [
    ["Field", "How to fill", "Dropdown options"],
    ["image", "Relative image path, e.g. test_images/0001.jpg.", "Free text"],
    ["scene", "Scene type of the image.", ", ".join(SCENES)],
    ["task", "Task type of this QA pair.", ", ".join(TASKS)],
    ["question", "Write a direct visual question. Keep it short and objective.", "Free text"],
    ["answer (recognition)", "Use Yes or No.", "Yes, No"],
    ["answer (counting)", "Use a number only, e.g. 0, 1, 2.", "Number only"],
    ["answer (localization)", "Bbox [x1,y1,x2,y2]. Origin at top-left corner.", "e.g. [338,70,664,371]"],
    ["target", "Object/concept name. Single form, lowercase with underscores.", ", ".join(TARGETS)],
    ["notes", "Format: annotator:X;confidence:level;tags.", "See dropdown for common combinations"],
    ["", "", ""],
    ["Target name guide (by scene)", "", ""],
    ["accident", "damaged_vehicle, overturned_vehicle, debris, traffic_accident", ""],
    ["construction", "traffic_cone, barricade, construction_vehicle, excavator", ""],
    ["person_vehicle", "car, pedestrian, motorcycle, bus, truck, vehicle", ""],
    ["jam", "vehicle, car, bus, truck", ""],
    ["firesmoke", "fire_source, smoke_plume, vehicle", ""],
    ["spill", "spill_area, vehicle", ""],
    ["weather", "snow_covered_road, flooded_road, vehicle", ""],
    ["", "", ""],
    ["Notes field format guide", "", ""],
    ["annotator:", "Who labeled this QA. Use A, B, C, etc.", ""],
    ["confidence:", "How confident: high, medium, or low.", ""],
    ["night_scene:true", "Add if the image was taken at night.", ""],
    ["partial_occlusion:true", "Add if the target is partially blocked.", ""],
    ["crowded:true", "Add if the scene is densely crowded.", ""],
    ["", "Example: annotator:A;confidence:medium;night_scene:true", ""],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create external traffic QA Excel template with dropdown menus."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--blank-rows", type=int, default=2000)
    return parser.parse_args()


def _add_validation(sheet, column: str, values: List[str], start_row: int, end_row: int) -> None:
    from openpyxl.worksheet.datavalidation import DataValidation

    # Excel data validation has a 255-char limit for the formula string.
    # If the list is too long, we truncate it and note it in the guide.
    formula = '"' + ",".join(values) + '"'
    if len(formula) > 255:
        # Truncate: keep as many items as fit within 250 chars
        truncated: List[str] = []
        for v in values:
            test = '"' + ",".join(truncated + [v]) + '"'
            if len(test) > 250:
                break
            truncated.append(v)
        formula = '"' + ",".join(truncated) + '"'
    validation = DataValidation(type="list", formula1=formula, allow_blank=True)
    validation.showDropDown = False  # allow free typing beyond dropdown options
    sheet.add_data_validation(validation)
    validation.add(f"{column}{start_row}:{column}{end_row}")


def main() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    args = parse_args()
    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "annotations"
    sheet.append(HEADERS)
    for row in EXAMPLE_ROWS:
        sheet.append(row)
    for _ in range(args.blank_rows):
        sheet.append(["", "", "", "", "", "", ""])

    # --- Styling ---
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Color-code the dropdown columns for visual clarity
    dropdown_fill = PatternFill("solid", fgColor="FFF2CC")  # light yellow
    for row_idx in range(2, args.blank_rows + len(EXAMPLE_ROWS) + 2):
        sheet[f"B{row_idx}"].fill = dropdown_fill  # scene
        sheet[f"C{row_idx}"].fill = dropdown_fill  # task
        sheet[f"F{row_idx}"].fill = dropdown_fill  # target
        sheet[f"G{row_idx}"].fill = dropdown_fill  # notes

    widths = [30, 16, 16, 72, 30, 26, 46]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "A2"

    total_rows = args.blank_rows + len(EXAMPLE_ROWS) + 1
    sheet.auto_filter.ref = f"A1:G{total_rows}"

    # --- Data validation dropdowns ---
    end_row = total_rows
    start_row = 2
    _add_validation(sheet, "B", SCENES, start_row, end_row)   # scene
    _add_validation(sheet, "C", TASKS, start_row, end_row)    # task
    _add_validation(sheet, "F", TARGETS, start_row, end_row)  # target
    _add_validation(sheet, "G", NOTES, start_row, end_row)    # notes

    # --- Guide sheet ---
    guide = workbook.create_sheet("guide")
    for row in GUIDE_ROWS:
        guide.append(row)
    guide.column_dimensions["A"].width = 28
    guide.column_dimensions["B"].width = 72
    guide.column_dimensions["C"].width = 48
    for cell in guide[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    # Bold the section headers
    section_headers = {
        "Target name guide (by scene)",
        "Notes field format guide",
    }
    for row in guide.iter_rows(min_row=2):
        if row[0].value in section_headers:
            for cell in row:
                cell.font = Font(bold=True)
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    workbook.save(args.output)
    print(f"Wrote {args.output}")
    print(f"  - {len(SCENES)} scene options")
    print(f"  - {len(TASKS)} task options")
    print(f"  - {len(TARGETS)} target options")
    print(f"  - {len(NOTES)} notes options")
    print(f"  - {args.blank_rows} blank rows ready for annotation")


if __name__ == "__main__":
    main()
