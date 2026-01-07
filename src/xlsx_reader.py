import math
import zipfile
import xml.etree.ElementTree as ET

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _load_shared_strings(zf: zipfile.ZipFile):
    try:
        with zf.open("xl/sharedStrings.xml") as f:
            root = ET.parse(f).getroot()
    except KeyError:
        return []

    strings = []
    for si in root.findall("x:si", NS):
        texts = [t.text or "" for t in si.findall(".//x:t", NS)]
        strings.append("".join(texts))
    return strings


def _parse_sheet(zf: zipfile.ZipFile, sheet_path: str):
    with zf.open(sheet_path) as f:
        root = ET.parse(f).getroot()

    sheet_data = root.find("x:sheetData", NS)
    rows = []
    for row in sheet_data.findall("x:row", NS):
        row_dict = {}
        for c in row.findall("x:c", NS):
            r = c.attrib.get("r")  # e.g., A1
            t = c.attrib.get("t")
            v = c.find("x:v", NS)
            if v is None:
                continue
            row_dict[r] = (t, v.text)
        rows.append(row_dict)
    return rows


def _col_to_index(col_ref: str) -> int:
    idx = 0
    for ch in col_ref:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _cell_col(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def read_xlsx_table(path: str, sheet_path: str = "xl/worksheets/sheet1.xml"):
    """Return headers (list[str]) and rows (list[list[float|None]])."""
    with zipfile.ZipFile(path) as zf:
        shared = _load_shared_strings(zf)
        rows = _parse_sheet(zf, sheet_path)

    if not rows:
        return [], []

    header_row = rows[0]
    col_name_by_index = {}
    for cell_ref, (t, v) in header_row.items():
        col = _cell_col(cell_ref)
        idx = _col_to_index(col)
        if t == "s":
            s_idx = int(v)
            name = shared[s_idx] if s_idx < len(shared) else f"str_{s_idx}"
        else:
            name = v
        col_name_by_index[idx] = name

    max_idx = max(col_name_by_index) if col_name_by_index else -1
    headers = [col_name_by_index.get(i, f"col_{i+1}") for i in range(max_idx + 1)]

    data_rows = []
    for row in rows[1:]:
        values = [None] * len(headers)
        for cell_ref, (t, v) in row.items():
            col = _cell_col(cell_ref)
            idx = _col_to_index(col)
            if idx >= len(headers):
                continue
            try:
                values[idx] = float(v)
            except (TypeError, ValueError):
                values[idx] = math.nan
        data_rows.append(values)

    return headers, data_rows
