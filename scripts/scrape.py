import json
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from loguru import logger
from lxml.html import HtmlMixin, fromstring, tostring
from tqdm import tqdm

# required by SEC policies
UA = "MinHtoo linmin.htoo@gmail.com"
session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

# Characters that often sneak in from SEC HTML
# - Zero-width: U+200B..U+200D, U+2060, U+FEFF
# - Soft hyphen: U+00AD
# - Non-breaking / thin / en space variants: U+00A0, U+202F, U+2000..U+200A, U+205F, U+3000
_INVISIBLES_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00AD]")
_UNISPACES_RE = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")

# for table parsing
_CAPTION_DENY_RE = re.compile(r"^(table of contents|index)$", re.I)
_CAPTION_KEYS_RE = re.compile(r"\b(table|schedule|summary|the following table|the table below)\b", re.I)
_CAPTION_UNITS_RE = re.compile(r"\((?:in|dollars|millions|thousands|unaudited)[^)]+\)", re.I)
_BULLET_CHARS = {"●", "•", "·", "▪", "–", "-", "■", "◦", "‣"}
_NUMERIC_RE = re.compile(r"[0-9]|[$,]|\((?:\d|[0-9,.$])+\)")  # crude but effective
_BORDER0_RE = re.compile(r"border\s*:\s*0", re.I)


# TODO: define proper return dataclass
def ticker_map(timeout: int = 30):
    # Official SEC static mapping of tickers <-> CIKs
    j = session.get("https://www.sec.gov/files/company_tickers.json", timeout=timeout).json()
    # normalize -> { "AAPL": "0000320193", ... }
    m = {}
    for _, row in j.items():
        cik = f"{int(row['cik_str']):010d}"
        m[row["ticker"].upper()] = cik
    return m


def list_10k_submissions(cik, max_docs: int = 2, timeout: int = 30):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    s = session.get(url, timeout=timeout).json()
    filings = s.get("filings", {}).get("recent", {})
    forms, acc, prim, dates = [filings.get(k, []) for k in ("form", "accessionNumber", "primaryDocument", "filingDate")]
    out = []
    for f, a, p, d in zip(forms, acc, prim, dates):
        # we only want 10Ks, ignore the rest
        if f == "10-K":
            out.append((a.replace("-", ""), p, d))
        if len(out) >= max_docs:
            break
    return out  # [(accession_no_nohyphens, primary_doc, filing_date), ...]


def download_primary(cik, acc_no, primary_doc, timeout: int = 60):
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/"
    url = urljoin(base, primary_doc)
    html = session.get(url, timeout=timeout).text
    return html, url


####################################
# TABLE CLASSIFICATION
####################################
def _table_shape(el) -> tuple[int, int, int]:
    rows = el.xpath(".//tr")
    r = len(rows)
    c = 0
    th = 0
    for tr in rows:
        tds = tr.xpath("./td|./th")
        c = max(c, len(tds))
        th += len(tr.xpath("./th"))
    return r, c, th


def _firstrow_cell_styles(el) -> list[str]:
    rows = el.xpath(".//tr")
    if not rows:
        return []
    return [(td.attrib.get("style", "") or "") for td in rows[0].xpath("./td|./th")]


def _table_text_cells(el) -> list[str]:
    return [_clean_inline_text(t) for t in el.xpath(".//td|.//th")]


def _is_tiny_width(style: str) -> bool:
    m = re.search(r"width\s*:\s*(\d+(?:\.\d+)?)\s*(pt|px)", style or "", flags=re.I)
    if not m:
        return False
    val, unit = float(m.group(1)), m.group(2).lower()
    # normalize px→pt roughly (1px≈0.75pt) – not critical, just be consistent
    if unit == "px":
        val *= 0.75
    return val <= 30.0


def _looks_like_bullet_list(el) -> bool:
    r, c, th = _table_shape(el)
    if th > 0:  # headers -> likely a real data table
        return False
    if r <= 2 and c <= 3:
        cells = _table_text_cells(el)
        if not cells:
            return False
        # any explicit bullet char? (also catches Wingdings bullet that gets normalized)
        has_bullet = any(
            (v in _BULLET_CHARS) or v.strip() in _BULLET_CHARS or v.strip().startswith(tuple(_BULLET_CHARS))
            for v in cells
        )
        # tiny first/second cells are classic vendor layout (e.g., width: 13.5pt)
        styles = _firstrow_cell_styles(el)
        tiny_cols = sum(1 for s in styles[:2] if _is_tiny_width(s))
        # almost no digits in the row text (lists rarely have numeric columns)
        digits_share = sum(bool(_NUMERIC_RE.search(t)) for t in cells) / max(len(cells), 1)
        return has_bullet and tiny_cols >= 1 and digits_share < 0.25
    return False


def _looks_like_layout_table(el) -> bool:
    # border:0 or role=presentation, no headers, few rows/cols
    r, c, th = _table_shape(el)
    style = el.attrib.get("style", "") or ""
    role = el.attrib.get("role", "") or ""
    cond_border0 = bool(_BORDER0_RE.search(style))
    return th == 0 and r <= 3 and c <= 3 and (cond_border0 or role.lower() == "presentation")


def _numeric_density(el) -> float:
    cells = _table_text_cells(el)
    if not cells:
        return 0.0
    return sum(bool(_NUMERIC_RE.search(t)) for t in cells) / len(cells)


def _classify_table(el) -> str:
    """
    Returns one of: 'data' | 'list' | 'layout' | 'unknown'
    """
    if _looks_like_bullet_list(el):
        return "list"
    if _looks_like_layout_table(el):
        return "layout"
    r, c, th = _table_shape(el)
    density = _numeric_density(el)
    # heuristics: headers, size, or numeric density → "data"
    if th > 0 or (r >= 3 and c >= 3) or density >= 0.35:
        return "data"
    # tiny one-row alignment tables usually aren't data
    if r <= 2 and c <= 2 and density < 0.2:
        return "layout"
    return "unknown"


def _similar_table_signature(el) -> tuple[int, int, int, bool]:
    r, c, th = _table_shape(el)
    style = el.attrib.get("style") or ""
    border0 = bool(_BORDER0_RE.search(style))
    return (c, th, int(border0), r >= 5)


def _gather_adjacent_data_tables(tables: list) -> list[list]:
    """
    Group consecutive 'data' tables that look similar (same columns/headers),
    with no big non-table blocks in between (aside from whitespace).
    """
    groups = []
    i = 0
    N = len(tables)
    while i < N:
        cur = tables[i]
        kind = _classify_table(cur)
        if kind != "data":
            groups.append([cur])  # non-data tables stay as singletons (will be skipped later)
            i += 1
            continue
        sig = _similar_table_signature(cur)
        group = [cur]
        j = i + 1
        while j < N:
            nxt = tables[j]
            if _classify_table(nxt) != "data":
                break
            # require same coarse signature; stop on big mismatch
            if _similar_table_signature(nxt) != sig:
                break
            # also ensure there isn't a heading/caption-like block inserted between them
            between = []
            node = group[-1]
            n = node.getnext()
            ok = True
            while n is not None and n is not nxt:
                # any 'blocky' captions / headings between → stop merging
                tag = (n.tag or "").lower()
                if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    ok = False
                    break
                # short caption-like paragraph between? stop.
                if tag in {"p", "div", "span"}:
                    txt = _clean_inline_text(n.text_content())
                    if _candidate_caption(txt, max_len=200):
                        ok = False
                        break
                n = n.getnext()
            if not ok:
                break
            group.append(nxt)
            j += 1
        groups.append(group)
        i = j
    return groups


####################################
# END OF TABLE CLASSIFICATION
####################################


def _normalize_ws(text: str) -> str:
    # collapse spaces/tabs, normalize newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u00A0", " ", text)  # &nbsp;
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_isolated_page_number(s: str) -> bool:
    # e.g., "99", "102", often a single line page header/footer number
    return bool(re.fullmatch(r"\d{1,4}", s))


def _drop_toc_headers(tree):
    """
    Remove elements that are exactly 'Table of Contents' (case-insensitive),
    plus a preceding isolated page number if present.
    """
    # Consider small block-ish nodes that often hold TOC headers
    candidates = tree.xpath("//p|//div|//span|//a|//h1|//h2|//h3|//h4|//h5|//h6")

    for el in list(candidates):
        txt = _clean_inline_text(el.text_content())
        if re.fullmatch(r"(?i)table\s+of\s+contents", txt):
            # remove preceding isolated page number if present
            prev = el.getprevious()
            if prev is not None:
                ptxt = _clean_inline_text(prev.text_content())
                if _is_isolated_page_number(ptxt):
                    parent = prev.getparent()
                    if parent is not None:
                        parent.remove(prev)
            # remove the TOC node itself
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def extract_text_without_tables_with_markers(html: str, doc_id: str) -> str:
    # TODO: fix lxml typing
    tree = fromstring(html.encode("utf-8"))
    _drop_toc_headers(tree)

    all_tables = tree.xpath("//table")
    # Replace tables: data → marker; list → convert to bullets; layout/unknown → drop
    for idx, t in enumerate(all_tables):
        kind = _classify_table(t)

        if kind == "data":
            table_id = f"{doc_id}::table::{idx:04d}"
            marker = f"[[TABLE::{table_id}]]"
            placeholder = fromstring(f"<p>{marker}</p>")
            t.addprevious(placeholder)
            t.drop_tree()

        elif kind == "list":
            # Turn the row(s) into plaintext bullets instead of a marker
            bullets = []
            for tr in t.xpath(".//tr"):
                cells = [_clean_inline_text(x.text_content()) for x in tr.xpath("./td|./th")]
                # The last cell usually holds the text
                if cells:
                    text = cells[-1]
                    if text:
                        bullets.append(f"- {text}")
            if bullets:
                block = fromstring("<div></div>")
                block.text = "\n".join(bullets)
                t.addprevious(block)
            t.drop_tree()

        else:  # 'layout' or 'unknown' → drop
            t.drop_tree()

    # Drop non-content and linearize (your existing logic)
    for bad in tree.xpath("//script|//style|//noscript|//iframe|//svg|//form|//nav|//header|//footer"):
        p = bad.getparent()
        if p is not None:
            p.remove(bad)

    for el in tree.xpath("//br"):
        el.drop_tag()
        el.tail = (el.tail or "") + "\n"
    for el in tree.xpath("//p|//div"):
        el.tail = (el.tail or "") + "\n\n"
    for el in tree.xpath("//li"):
        el.text = el.text or ""
        if not el.text.startswith("- "):
            el.text = "- " + el.text
        el.tail = (el.tail or "") + "\n"
    for el in tree.xpath("//h1|//h2|//h3|//h4|//h5|//h6"):
        el.tail = (el.tail or "") + "\n\n"

    text = tree.text_content()
    text = _normalize_ws(text)
    text = clean_invisible_and_blank_lines(text, keep_blank=1)
    return text


def clean_invisible_and_blank_lines(text: str, keep_blank: int = 1) -> str:
    # 1) Remove zero-width & formatting chars entirely
    text = _INVISIBLES_RE.sub("", text)

    # 2) Normalize odd Unicode spaces to a plain space
    text = _UNISPACES_RE.sub(" ", text)

    # 3) Trim trailing spaces on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)

    # 4) Collapse runs of lines that are empty (after normalization)
    # keep_blank=1 -> at most one empty line between blocks
    # Convert multiple blank lines to exactly keep_blank
    if keep_blank >= 0:
        # First, normalize any lines containing only whitespace to just "\n"
        text = re.sub(r"(?m)^[ \t]*\n", "\n", text)
        # Then collapse runs of newlines
        limit = keep_blank + 1  # e.g., keep 1 blank -> at most two consecutive "\n"
        text = re.sub(r"\n{%d,}" % (limit + 1), "\n" * limit, text)

    return text.strip()


def _clean_cell(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = _INVISIBLES_RE.sub("", s)
    s = _UNISPACES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # normalize standalone dashes/empties often used for zeros/NA
    return s


def _looks_like_all(s: pd.Series, pattern: str) -> bool:
    # True if every non-empty cell matches pattern
    non_empty = s.astype(str).str.strip().replace({"nan": ""})
    non_empty = non_empty[non_empty != ""]
    if non_empty.empty:
        return False
    return non_empty.str.fullmatch(pattern).all()


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join([_clean_cell(c) for c in tup if _clean_cell(c)]) for tup in df.columns.values]
    else:
        df.columns = [_clean_cell(c) for c in df.columns]
    return df


def _drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    df.columns = [c if c else f"col_{i}" for i, c in enumerate(df.columns)]
    # Drop columns that are entirely empty
    empty_cols = [c for c in df.columns if df[c].astype(str).str.strip().replace("nan", "").eq("").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    # Drop rows that are entirely empty after cleaning
    df = df[~df.apply(lambda r: all((_clean_cell(v) == "") for v in r), axis=1)]
    return df


def _normalize_numeric_series(s: pd.Series) -> pd.Series:
    def norm(v: Any) -> float | str | None:
        t = _clean_cell(v)
        if t in {"–", "-", ""}:
            return None
        # parentheses for negatives
        neg = False
        if re.fullmatch(r"\((.+)\)", t):
            neg = True
            t = re.sub(r"^\((.+)\)$", r"\1", t)
        # strip currency/percent and commas
        t2 = re.sub(r"[$,%]", "", t)
        t2 = t2.replace(",", "")
        # some EPS like "3.74" is fine, keep as float
        try:
            val = float(t2)
            return -val if neg else val
        except ValueError:
            return t  # leave textual cells alone (row labels)

    return s.map(norm)


def _merge_unit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    SEC tables often have patterns like:
        [ '$' ][ 1,619.2 ][ '' ][ 19.7 ][ '%' ] ...
    We:
        - drop columns that are purely '$'
        - if a column of '%' follows a numeric column, mark that numeric column's name with ' (%)' and drop the '%' col
    """
    df = df.copy()
    cols = list(df.columns)
    keep = [True] * len(cols)
    new_names = cols[:]

    i = 0
    while i < len(cols):
        col = cols[i]
        series = df[col]

        # Drop pure '$' columns adjacent to numeric columns
        if _looks_like_all(series, r"\$"):
            keep[i] = False
            i += 1
            continue

        # If next column is purely '%' markers, merge units into current
        if i + 1 < len(cols) and _looks_like_all(df[cols[i + 1]], r"%+"):
            # decorate current column name with ' (%)'
            base = new_names[i]
            if not base.endswith("(%)"):
                new_names[i] = (base + " (%)").strip()
            keep[i + 1] = False
            i += 2
            continue

        # If current column is purely '%' and previous was value, mark previous
        if _looks_like_all(series, r"%+") and i - 1 >= 0 and keep[i - 1]:
            base = new_names[i - 1]
            if not base.endswith("(%)"):
                new_names[i - 1] = (base + " (%)").strip()
            keep[i] = False
            i += 1
            continue

        i += 1

    df = df.loc[:, [c for c, k in zip(cols, keep) if k]]
    df.columns = [n for n, k in zip(new_names, keep) if k]
    return df


def _clean_inline_text(s: HtmlMixin | str) -> str:
    # NEW: accept lxml elements too
    if isinstance(s, HtmlMixin):
        s = s.text_content()
    s = s or ""
    s = _INVISIBLES_RE.sub("", s)
    s = _UNISPACES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _candidate_caption(text: str, max_len: int = 500) -> bool:
    if not text or _CAPTION_DENY_RE.search(text):
        return False
    # A “caption-like” line is short and either:
    # - has table-ish keywords, or
    # - ends with a colon, or
    # - includes unit hints ( (in millions), (unaudited), etc. )
    if len(text) <= max_len and (_CAPTION_KEYS_RE.search(text) or text.endswith(":") or _CAPTION_UNITS_RE.search(text)):
        return True
    return False


def _infer_table_caption(el, max_up: int = 2, max_back: int = 12, max_length: int = 500) -> str:
    """
    Prefer <caption>. Otherwise scan a few previous siblings (and ancestors' previous siblings)
    for a short <p>/<div>/<span> line that looks like a caption. Skip "Table of Contents" etc.
    Also pull in a nearby units line if present between the caption and the table.
    """
    # 1) Real <caption> tag if present
    caps = el.xpath(".//caption")
    if caps:
        cap = _clean_inline_text(caps[0].text_content())
        if cap and not _CAPTION_DENY_RE.search(cap):
            return cap[:max_length]

    # Utility: iterate previous siblings of a node
    def _prev_siblings(node):
        cur = node.getprevious()
        while cur is not None:
            yield cur
            cur = cur.getprevious()

    chosen_text = ""

    # 2) Walk up to a couple ancestors; on each level, scan up to N previous siblings
    node = el
    for _ in range(max_up + 1):
        back = 0
        for prev in _prev_siblings(node):
            if back >= max_back:
                break
            back += 1

            # Only consider small block-ish elements (p/div/span)
            tag = (prev.tag or "").lower()
            if tag not in {"p", "div", "span"}:
                continue

            text = _clean_inline_text(prev.text_content())
            if _candidate_caption(text, max_len=max_length):
                chosen_text = text
                # 2a) Try to append a nearby units line that sits between caption and table
                # (e.g. "(in millions)" on the very next <p> above the table)
                # Scan forward from prev towards el:
                nxt = prev.getnext()
                while nxt is not None and nxt is not el:
                    t2 = _clean_inline_text(nxt.text_content())
                    if _CAPTION_UNITS_RE.search(t2) and len(t2) <= 120:
                        # Avoid duplicating units already in chosen_text
                        if t2 not in chosen_text:
                            chosen_text = f"{chosen_text} {t2}"
                        break
                    nxt = nxt.getnext()
                return chosen_text[:max_length]
        node = node.getparent()
        if node is None:
            break

    # 3) Fallback: look for a nearby heading that itself looks like a caption
    node = el
    for _ in range(max_up + 1):
        for prev in _prev_siblings(node):
            tag = (prev.tag or "").lower()
            if re.fullmatch(r"h[1-6]", tag, re.I):
                text = _clean_inline_text(prev.text_content())
                if _candidate_caption(text, max_len=max_length):
                    return text[:max_length]
        node = node.getparent()
        if node is None:
            break

    return ""


def extract_tables_markdown_and_facts(html_bytes: bytes, max_length: int = 500) -> list[dict[str, Any]]:
    tree = fromstring(html_bytes)
    tables = tree.xpath("//table")
    groups = _gather_adjacent_data_tables(tables)  # merge candidates

    out = []
    for group in groups:
        # Only keep groups whose first member is 'data'
        if _classify_table(group[0]) != "data":
            continue

        # Build one HTML fragment by concatenating the tables' rows
        frags = [tostring(t, encoding="unicode") for t in group]
        # Wrap in a single table so pandas reads once
        merged_html = (
            "<table>" + "".join(re.sub(r"(?is)^<table[^>]*>|</table>$", "", f).strip() for f in frags) + "</table>"
        )

        try:
            dfs = pd.read_html(StringIO(merged_html), flavor="lxml")
        except Exception:
            try:
                dfs = pd.read_html(StringIO(merged_html), flavor="bs4")
            except Exception:
                continue

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True, sort=False)
        df = _flatten_multiindex_columns(df).map(_clean_cell)
        df = _drop_empty_rows_cols(df)
        df = _merge_unit_columns(df)
        if df.empty or df.shape[1] < 2:
            continue

        row_label_col = df.columns[0]
        for c in df.columns:
            if c != row_label_col:
                df[c] = _normalize_numeric_series(df[c])
        df = _drop_empty_rows_cols(df)
        if df.empty or df.shape[1] < 2:
            continue

        md = df.to_markdown(index=False)
        # Caption inference only runs once per merged group (use the first table element)
        caption = _infer_table_caption(group[0], max_length=max_length)

        facts = []
        for _, row in df.iterrows():
            row_label = row.get(row_label_col, "")
            if not row_label:
                continue
            for col in df.columns:
                if col == row_label_col:
                    continue
                val = row.get(col, "")
                if val == "":
                    continue
                facts.append(f"{caption}: {row_label} — {col} = {val}" if caption else f"{row_label} — {col} = {val}")

        out.append(
            {
                "caption": caption,
                "markdown": md,
                "facts": facts,
                "df": df,  # keep for debugging as you do now
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
            }
        )
    return out


def fetch_10ks_for_tickers(tickers: list[str], output_dir: Path, per_company: int = 2, delay: float = 0.2):
    out_raw_folder = output_dir / "10k_raw"
    out_raw_folder.mkdir(parents=True, exist_ok=True)

    t2c = ticker_map()
    for t in tqdm(tickers, desc="fetching tickers"):
        cik = t2c[t.upper()]
        for acc_no, primary, fdate in tqdm(list_10k_submissions(cik, per_company), desc=f"processing ticker {t}"):
            base = f"{t.upper()}_{acc_no}"
            doc_id = f"10k::{t.upper()}::{acc_no}"

            # fetch raw HTML
            html, src = download_primary(cik, acc_no, primary)

            # first, extract table markdown + caption + facts
            tables = extract_tables_markdown_and_facts(html.encode("utf-8"))

            # then, extract text with table markers in place of table content
            # dont include caption in table marker bcos caption already exists in `text` nearby
            text = extract_text_without_tables_with_markers(
                html,
                doc_id,
            )

            # finally, write to disk
            norm_dir = out_dir / "normalized" / base
            norm_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "doc_id": doc_id,
                "ticker": t.upper(),
                "cik": cik,
                "filing_date": fdate,
                "accession": acc_no,
                "primary": primary,
                "source_url": src,
                "form": "10-K",
            }
            (norm_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

            # text.jsonl (one record)
            # TODO: upgrade later to multiple section records after adding sectionization (heading parsing like Risks)
            with (norm_dir / "text.jsonl").open("w", encoding="utf-8") as f:
                rec = {
                    "doc_id": doc_id,
                    "section_path": [],  # TODO: update after sectionization
                    "text": text,
                    "span": None,
                    "source_url": src,
                    "ticker": t.upper(),
                    "cik": cik,
                    "filing_date": fdate,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # write table data
            tables_md = []
            facts_rows = []

            for i, tb in enumerate(tables):
                table_id = f"{doc_id}::table::{i:04d}"
                caption = tb["caption"]

                tables_md.append(
                    {
                        "doc_id": doc_id,
                        "table_id": table_id,
                        "caption": caption,
                        "markdown": tb["markdown"],
                        "n_rows": tb["n_rows"],
                        "n_cols": tb["n_cols"],
                        "section_path": [],  # TODO: update after sectionization
                        "source_url": src,
                        "units_hint": None,
                        "order_index": i,
                    }
                )

                for fact in tb["facts"]:
                    facts_rows.append(
                        {
                            "doc_id": doc_id,
                            "table_id": table_id,
                            "fact": fact,
                            "order_index": i,
                        }
                    )

            # write jsonl
            with (norm_dir / "tables_md.jsonl").open("w", encoding="utf-8") as f:
                for r in tables_md:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with (norm_dir / "facts.jsonl").open("w", encoding="utf-8") as f:
                for r in facts_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # TODO: consider parquet for more structured data
            # pq.write_table(pa.Table.from_pylist(tbl_rows), norm_dir / "tables.parquet")

            # write raw data
            (out_raw_folder / f"{base}.html").write_text(html, encoding="utf-8")
            (out_raw_folder / f"{base}.txt").write_text(text, encoding="utf-8")

            # be polite to SEC servers
            time.sleep(delay)

    logger.success("Done.")


if __name__ == "__main__":
    # TODO: get list of tickers from somewhere, e.g. top 500 companies
    tickers = ["APH", "GOOGL", "NVDA"]
    # tickers = ["APH"]
    out_dir = Path("./data")
    fetch_10ks_for_tickers(tickers, out_dir, per_company=5, delay=0.1)
