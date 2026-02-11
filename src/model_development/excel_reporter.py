"""
Excel Reporter

Generates a single Excel workbook with all elimination and evaluation details.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows


logger = logging.getLogger(__name__)


# Styles
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
KEPT_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
ELIM_FILL = PatternFill(start_color="FCE4EC", end_color="FCE4EC", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin'),
)


def generate_report(
    output_path: str,
    summary: Dict[str, Any],
    elimination_results: List[Any],
    corr_pairs_df: Optional[pd.DataFrame],
    selection_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    lift_tables: Dict[str, pd.DataFrame],
    importance_df: pd.DataFrame,
) -> str:
    """
    Generate the full Excel report.

    Args:
        output_path: Path for the output Excel file.
        summary: Dict of summary key-value pairs.
        elimination_results: List of EliminationResult objects.
        corr_pairs_df: Correlation pairs details DataFrame.
        selection_df: Sequential selection details DataFrame.
        performance_df: Quarterly performance DataFrame.
        lift_tables: Dict of period -> lift table DataFrame.
        importance_df: Feature importance DataFrame.

    Returns:
        Path to the generated Excel file.
    """
    wb = Workbook()

    # 00_Summary
    _write_summary_sheet(wb, summary)

    # Elimination sheets
    for result in elimination_results:
        _write_df_sheet(wb, result.step_name, result.details_df)

    # 05_Correlation_Matrix
    if corr_pairs_df is not None and len(corr_pairs_df) > 0:
        _write_df_sheet(wb, "05_Corr_Pairs", corr_pairs_df)

    # 06_Sequential_Selection
    _write_df_sheet(wb, "06_Selection", selection_df)

    # 07_Model_Performance
    _write_df_sheet(wb, "07_Performance", performance_df)

    # 07_Lift_Tables (one sheet per period, or combined)
    _write_lift_sheets(wb, lift_tables)

    # 07_Feature_Importance
    _write_df_sheet(wb, "07_Importance", importance_df)

    # Remove default empty sheet if exists
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info(f"COMPLETE | Excel saved: {output_path}")
    return output_path


def _write_summary_sheet(wb: Workbook, summary: Dict[str, Any]) -> None:
    """Write the 00_Summary sheet."""
    ws = wb.create_sheet("00_Summary")

    ws.column_dimensions['A'].width = 35
    ws.column_dimensions['B'].width = 50

    # Title
    ws['A1'] = "Credit Scoring Model Development Report"
    ws['A1'].font = Font(bold=True, size=14, color="2F5496")
    ws.merge_cells('A1:B1')

    row = 3
    for key, value in summary.items():
        cell_a = ws.cell(row=row, column=1, value=key)
        cell_b = ws.cell(row=row, column=2, value=str(value))
        cell_a.font = Font(bold=True)
        cell_a.border = THIN_BORDER
        cell_b.border = THIN_BORDER
        row += 1


def _write_df_sheet(
    wb: Workbook, sheet_name: str, df: pd.DataFrame
) -> None:
    """Write a DataFrame to a styled sheet."""
    if df is None or len(df) == 0:
        ws = wb.create_sheet(sheet_name)
        ws['A1'] = "No data"
        return

    ws = wb.create_sheet(sheet_name)

    # Write headers
    for col_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center')
        cell.border = THIN_BORDER

    # Write data rows
    for row_idx, (_, row_data) in enumerate(df.iterrows(), 2):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = THIN_BORDER

            # Handle numpy/pandas types
            if pd.isna(value):
                cell.value = None
            elif isinstance(value, (np.integer,)):
                cell.value = int(value)
            elif isinstance(value, (np.floating,)):
                cell.value = float(value)
            else:
                cell.value = value

        # Color code rows by Status column if present
        if 'Status' in df.columns:
            status_col = list(df.columns).index('Status') + 1
            status_val = ws.cell(row=row_idx, column=status_col).value
            fill = None
            if status_val == 'Eliminated':
                fill = ELIM_FILL
            elif status_val == 'Kept' or status_val == 'Added':
                fill = KEPT_FILL

            if fill:
                for col_idx in range(1, len(df.columns) + 1):
                    ws.cell(row=row_idx, column=col_idx).fill = fill

    # Auto-fit column widths (approximate)
    for col_idx, col_name in enumerate(df.columns, 1):
        max_len = len(str(col_name))
        for row_idx in range(2, min(len(df) + 2, 102)):  # sample first 100 rows
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[
            ws.cell(row=1, column=col_idx).column_letter
        ].width = min(max_len + 3, 40)

    # Freeze header row
    ws.freeze_panes = 'A2'

    # Auto-filter
    ws.auto_filter.ref = ws.dimensions


def _write_lift_sheets(
    wb: Workbook, lift_tables: Dict[str, pd.DataFrame]
) -> None:
    """Write combined lift table sheet."""
    if not lift_tables:
        return

    combined_rows = []
    for period, lt in lift_tables.items():
        lt_copy = lt.copy()
        lt_copy.insert(0, 'Period', period)
        combined_rows.append(lt_copy)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        _write_df_sheet(wb, "07_Lift_Tables", combined)
