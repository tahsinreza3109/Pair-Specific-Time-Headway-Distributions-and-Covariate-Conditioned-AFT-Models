from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from artifact_tool import Blob, SpreadsheetFile, Workbook

# =============================================================================
# USER PATHS
# =============================================================================
BASE = Path(r"D:\Headway")

T5_PATH = BASE / "Tables" / "T5_41_Sequential_AFT_Selection_VIF.xlsx"
T6_PATH = BASE / "Tables" / "T6_Final_AFT_Bootstrap_CoxSnell.xlsx"
DATA_PATH = BASE / "data3.xlsx"

OUTPUT_XLSX = BASE / "Tables" / "Table8_Final_AFT_Models.xlsx"
OUTPUT_CSV = BASE / "Tables" / "Table8_Final_AFT_Models.csv"

# For testing in another environment, replace the paths above directly.

PAIR_ORDER = [
    "BTW_following_4W",
    "BTW_following_MT_3W",
    "BTW_following_NMT_3W",
    "BTW_following_MT_2W",
    "BTW_following_NMT_2W",
    "PR_following_4W",
    "PR_following_MT_3W",
    "PR_following_NMT_3W",
]

COVARIATE_LABELS = {
    "Speed_Difference": "Speed difference",
    "Target_Speed_km/hr": "Target speed",
    "Leading_Speed_km/hr": "Leader speed",
    "Off_centeredness": "Off-centeredness",
    "Occupancy": "Occupancy",
    "Flow_pcu/hr/m": "Traffic flow",
    "Site": "Site",
}

# =============================================================================
# HELPERS
# =============================================================================
def read_sheet_records(path: Path, sheet_name: str) -> List[Dict[str, Any]]:
    """Read one Excel sheet into a list of dictionaries using artifact_tool."""
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    wb = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sheet = wb.worksheets.get_item(sheet_name)
    values = sheet.get_range("A1").get_current_region().values

    if not values or len(values) < 2:
        raise ValueError(f"Sheet {sheet_name!r} in {path.name} is empty.")

    headers = [str(value).strip() if value is not None else "" for value in values[0]]
    records: List[Dict[str, Any]] = []

    for row in values[1:]:
        padded = list(row) + [None] * (len(headers) - len(row))
        if all(value is None or str(value).strip() == "" for value in padded):
            continue
        records.append(dict(zip(headers, padded[: len(headers)])))

    return records


def require_columns(records: Sequence[Mapping[str, Any]], required: Iterable[str], source: str) -> None:
    if not records:
        raise ValueError(f"No records were read from {source}.")
    available = set(records[0].keys())
    missing = [column for column in required if column not in available]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}. Available: {sorted(available)}")


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "nan", "None"}


def as_float(value: Any, label: str) -> float:
    if is_blank(value):
        raise ValueError(f"Missing numeric value for {label}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {label}: {value}")
    return result


def as_int(value: Any, label: str) -> int:
    result = int(round(as_float(value, label)))
    return result


def pair_label(pair: str, data_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = [row for row in data_rows if row["Pair"] == pair]
    if not rows:
        raise ValueError(f"Pair {pair!r} was not found in data3.xlsx.")

    targets = {str(row["V_Target"]).strip() for row in rows}
    leaders = {str(row["V_Leading_Class"]).strip() for row in rows}
    if len(targets) != 1 or len(leaders) != 1:
        raise ValueError(
            f"Pair {pair!r} maps to multiple classes in data3.xlsx: "
            f"targets={targets}, leaders={leaders}"
        )
    return f"{next(iter(targets))} → {next(iter(leaders))}"


def covariate_label(value: Any) -> str:
    if is_blank(value) or str(value).strip() == "(none)":
        return "None"
    covariates = [part.strip() for part in str(value).split(",") if part.strip()]
    return "; ".join(COVARIATE_LABELS.get(covariate, covariate) for covariate in covariates)


def find_unique(
    records: Sequence[Mapping[str, Any]],
    *,
    pair: str,
    distribution: str,
    source: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in records
        if str(row.get("Pair", "")).strip() == pair
        and str(row.get("Distribution", "")).strip() == distribution
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for pair={pair!r}, distribution={distribution!r} "
            f"in {source}; found {len(matches)}."
        )
    return matches[0]


def stat_string(stat: Any, p_value: Any, bootstrap_status: Any) -> str:
    status = "" if bootstrap_status is None else str(bootstrap_status).strip()
    if is_blank(p_value):
        if status and status != "OK":
            return "Bootstrap invalid"
        return "Not available"
    return f"{as_float(stat, 'test statistic'):.3f} ({as_float(p_value, 'bootstrap p'):.3f})"


def decision_label(decision_row: Mapping[str, Any]) -> str:
    selected = decision_row.get("Selected distribution")
    if is_blank(selected):
        return "Not adequate"

    selected_name = str(selected).strip()
    lowest_name = str(decision_row.get("Lowest-AIC tested distribution", "")).strip()
    if selected_name != lowest_name:
        return "Accepted alternative"
    return "Accepted"


def count_pair_rows(data_rows: Sequence[Mapping[str, Any]], pair: str) -> int:
    return sum(1 for row in data_rows if row["Pair"] == pair)


# =============================================================================
# BUILD TABLE 8
# =============================================================================
def build_table8(
    t5_path: Path,
    t6_path: Path,
    data_path: Path,
) -> tuple[List[List[Any]], List[List[Any]]]:
    t5_models = read_sheet_records(t5_path, "S1_Model_summary")
    t6_gof = read_sheet_records(t6_path, "S3_AFT_final_GOF")
    t6_decisions = read_sheet_records(t6_path, "S5_Final_decisions")
    data_rows = read_sheet_records(data_path, "Sheet1")

    require_columns(
        t5_models,
        ["Pair", "Distribution", "n", "Baseline AIC", "Final AIC", "Total ΔAIC", "Final covariates"],
        "T5/S1_Model_summary",
    )
    require_columns(
        t6_gof,
        [
            "Pair", "Distribution", "Final covariates", "n", "AIC", "ΔAIC",
            "KS D", "KS bootstrap p", "AD A²", "AD bootstrap p", "Bootstrap status",
        ],
        "T6/S3_AFT_final_GOF",
    )
    require_columns(
        t6_decisions,
        [
            "Pair", "Selected distribution", "Selected covariates",
            "Lowest-AIC tested distribution", "Lowest-AIC test outcome",
        ],
        "T6/S5_Final_decisions",
    )
    require_columns(data_rows, ["V_Target", "V_Leading_Class", "Pair"], "data3/Sheet1")

    decisions_by_pair = {str(row["Pair"]).strip(): row for row in t6_decisions}

    table_header = [
        "Target → leader",
        "Selected distribution",
        "Retained covariates",
        "AIC improvement over covariate-free model",
        "Final AIC",
        "KS D (p)",
        "AD A² (p)",
        "Decision",
    ]
    table_rows: List[List[Any]] = [table_header]

    audit_header = [
        "Pair key",
        "Target → leader",
        "n from data3",
        "n in T5",
        "n in T6",
        "Displayed distribution",
        "T6 selected distribution",
        "Lowest-AIC tested distribution",
        "T5 baseline AIC",
        "T5 final AIC",
        "T5 total ΔAIC",
        "T6 final AIC",
        "T6 ΔAIC within final evidence set",
        "Bootstrap status",
        "Validation",
    ]
    audit_rows: List[List[Any]] = [audit_header]

    for pair in PAIR_ORDER:
        if pair not in decisions_by_pair:
            raise KeyError(f"Pair {pair!r} is absent from T6/S5_Final_decisions.")

        decision = decisions_by_pair[pair]
        selected_distribution = decision.get("Selected distribution")
        lowest_distribution = decision.get("Lowest-AIC tested distribution")

        displayed_distribution = (
            str(selected_distribution).strip()
            if not is_blank(selected_distribution)
            else str(lowest_distribution).strip()
        )

        t6_row = find_unique(
            t6_gof,
            pair=pair,
            distribution=displayed_distribution,
            source="T6/S3_AFT_final_GOF",
        )
        t5_row = find_unique(
            t5_models,
            pair=pair,
            distribution=displayed_distribution,
            source="T5/S1_Model_summary",
        )

        n_data = count_pair_rows(data_rows, pair)
        n_t5 = as_int(t5_row["n"], f"T5 n for {pair}")
        n_t6 = as_int(t6_row["n"], f"T6 n for {pair}")
        validation = "OK" if n_data == n_t5 == n_t6 else "N MISMATCH"
        if validation != "OK":
            raise ValueError(
                f"Sample-size mismatch for {pair}: data3={n_data}, T5={n_t5}, T6={n_t6}."
            )

        t5_improvement = as_float(t5_row["Total ΔAIC"], f"T5 Total ΔAIC for {pair}")
        t5_recalculated = (
            as_float(t5_row["Baseline AIC"], f"T5 Baseline AIC for {pair}")
            - as_float(t5_row["Final AIC"], f"T5 Final AIC for {pair}")
        )
        if not math.isclose(t5_improvement, t5_recalculated, rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                f"AIC-improvement inconsistency for {pair}, {displayed_distribution}: "
                f"stored={t5_improvement}, recalculated={t5_recalculated}."
            )

        bootstrap_status = t6_row.get("Bootstrap status")
        retained_covariates = t6_row.get("Final covariates")

        table_rows.append(
            [
                pair_label(pair, data_rows),
                displayed_distribution,
                covariate_label(retained_covariates),
                round(t5_improvement, 3),
                round(as_float(t6_row["AIC"], f"T6 AIC for {pair}"), 3),
                stat_string(t6_row["KS D"], t6_row["KS bootstrap p"], bootstrap_status),
                stat_string(t6_row["AD A²"], t6_row["AD bootstrap p"], bootstrap_status),
                decision_label(decision),
            ]
        )

        audit_rows.append(
            [
                pair,
                pair_label(pair, data_rows),
                n_data,
                n_t5,
                n_t6,
                displayed_distribution,
                None if is_blank(selected_distribution) else str(selected_distribution).strip(),
                str(lowest_distribution).strip(),
                as_float(t5_row["Baseline AIC"], f"T5 Baseline AIC for {pair}"),
                as_float(t5_row["Final AIC"], f"T5 Final AIC for {pair}"),
                t5_improvement,
                as_float(t6_row["AIC"], f"T6 AIC for {pair}"),
                as_float(t6_row["ΔAIC"], f"T6 ΔAIC for {pair}"),
                bootstrap_status,
                validation,
            ]
        )

    return table_rows, audit_rows


# =============================================================================
# EXPORT
# =============================================================================
def export_table8(table_rows: List[List[Any]], audit_rows: List[List[Any]]) -> None:
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook.create()
    sheet = wb.worksheets.add("Table 8")

    # Title and table
    sheet.merge_cells("A1:H1")
    sheet.get_range("A1").values = [["Table 8. Final Covariate-Conditioned AFT Models by Target–Leader Pair"]]
    sheet.get_range("A3:H11").values = table_rows

    # Manuscript-style note
    note = (
        "Note: AIC improvement is calculated against the covariate-free model of the same "
        "distribution using T5. Final AIC and bootstrap goodness-of-fit results are taken from "
        "T6. The sample size for every pair is independently verified against data3. "
        "Acceptance requires bootstrap KS p ≥ 0.05 and AD p ≥ 0.05. For unresolved pairs, "
        "the lowest-AIC tested distribution is displayed and marked Not adequate."
    )
    sheet.merge_cells("A13:H15")
    sheet.get_range("A13").values = [[note]]

    # Formatting
    sheet.get_range("A1:H1").format = {
        "fill": "#1F4E78",
        "font": {"bold": True, "color": "#FFFFFF", "size": 13},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "row_height": 30,
    }
    sheet.get_range("A3:H3").format = {
        "fill": "#D9EAF7",
        "font": {"bold": True, "color": "#000000"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "row_height": 34,
        "borders": {
            "top": {"style": "thin", "color": "#7F7F7F"},
            "bottom": {"style": "thin", "color": "#7F7F7F"},
            "left": {"style": "thin", "color": "#7F7F7F"},
            "right": {"style": "thin", "color": "#7F7F7F"},
        },
    }
    sheet.get_range("A4:H11").format = {
        "vertical_alignment": "center",
        "wrap_text": True,
        "borders": {
            "top": {"style": "hair", "color": "#B7B7B7"},
            "bottom": {"style": "hair", "color": "#B7B7B7"},
            "left": {"style": "hair", "color": "#B7B7B7"},
            "right": {"style": "hair", "color": "#B7B7B7"},
        },
    }
    sheet.get_range("D4:E11").format.number_format = "0.000"
    sheet.get_range("A13:H15").format = {
        "font": {"italic": True, "size": 9},
        "wrap_text": True,
        "vertical_alignment": "top",
        "fill": "#F7F7F7",
    }

    # Highlight decisions
    decision_range = sheet.get_range("H4:H11")
    decision_range.conditional_formats.add_custom(
        '=H4="Accepted"',
        {"fill": "#E2F0D9", "font": {"color": "#375623", "bold": True}},
    )
    decision_range.conditional_formats.add_custom(
        '=H4="Accepted alternative"',
        {"fill": "#FFF2CC", "font": {"color": "#7F6000", "bold": True}},
    )
    decision_range.conditional_formats.add_custom(
        '=H4="Not adequate"',
        {"fill": "#FCE4D6", "font": {"color": "#9C0006", "bold": True}},
    )

    # Practical column widths
    widths = {
        "A:A": 18,
        "B:B": 21,
        "C:C": 38,
        "D:D": 19,
        "E:E": 12,
        "F:G": 16,
        "H:H": 21,
    }
    for column_range, width in widths.items():
        sheet.get_range(column_range).format.column_width = width
    sheet.get_range("A4:H11").format.row_height = 28
    sheet.freeze_panes.freeze_rows(3)

    # Audit sheet
    audit = wb.worksheets.add("Audit")
    rows = len(audit_rows)
    cols = len(audit_rows[0])
    audit.get_range_by_indexes(0, 0, rows, cols).values = audit_rows
    audit.get_range_by_indexes(0, 0, 1, cols).format = {
        "fill": "#5B9BD5",
        "font": {"bold": True, "color": "#FFFFFF"},
        "wrap_text": True,
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
    }
    audit.get_range_by_indexes(1, 0, rows - 1, cols).format.wrap_text = True
    audit.get_range("I2:M9").format.number_format = "0.000000"
    audit.get_range("A1:O9").format.autofit_columns()
    # Cap text-heavy columns after autofit.
    audit.get_range("A:A").format.column_width = 24
    audit.get_range("B:B").format.column_width = 18
    audit.get_range("F:H").format.column_width = 24
    audit.get_range("N:N").format.column_width = 38
    audit.freeze_panes.freeze_rows(1)

    SpreadsheetFile.export_xlsx(wb).save(str(OUTPUT_XLSX))

    # CSV contains the exact manuscript table only.
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerows(table_rows)

    print("Generated:")
    print(OUTPUT_XLSX)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    table8, audit = build_table8(T5_PATH, T6_PATH, DATA_PATH)
    export_table8(table8, audit)
