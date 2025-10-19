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
from lxml.html import fromstring, tostring
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
_CAPTION_DENY_RE   = re.compile(r"^(table of contents|index)$", re.I)
_CAPTION_KEYS_RE   = re.compile(r"\b(table|schedule|summary|the following table|the table below)\b", re.I)
_CAPTION_UNITS_RE  = re.compile(r"\((?:in|dollars|millions|thousands|unaudited)[^)]+\)", re.I)


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


def strip_html_to_text(html):
    """
    Deprecated as it is worse than lxml parsing.
    """
    # Remove script/style sections
    # FIXED: </\1> instead of </\\1>
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    # Replace <br> with newline
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    # Replace </p> with double newline
    text = re.sub(r"(?is)</p>", "\n\n", text)
    # Remove all other HTML tags
    text = re.sub(r"(?is)<.*?>", " ", text)
    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse excessive newlines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _normalize_ws(text: str) -> str:
    # collapse spaces/tabs, normalize newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u00A0", " ", text)  # &nbsp;
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_lxml_preserve_tables(html: str) -> str:
    tree = fromstring(html.encode("utf-8"))

    # Drop obvious non-content
    for bad in tree.xpath("//script|//style|//noscript|//iframe|//svg|//form|//nav|//header|//footer"):
        bad.getparent().remove(bad)

    # Insert structural separators so text_content() doesn't mush everything together
    # Paragraphs / breaks
    for el in tree.xpath("//br"):
        el.drop_tag()  # keep tail
        if el.tail is None:
            el.tail = "\n"
        else:
            el.tail = el.tail + "\n"

    for el in tree.xpath("//p|//div"):
        # add blank line after paragraphs / blocks
        el.tail = (el.tail or "") + "\n\n"

    # lists
    for el in tree.xpath("//li"):
        el.text = el.text or ""
        # bullet-ish prefix helps readability in plain text
        el.text = ("- " + el.text) if not el.text.startswith("- ") else el.text
        el.tail = (el.tail or "") + "\n"

    # Tables: separate cells with tabs and rows with newlines
    for el in tree.xpath("//td|//th"):
        el.tail = (el.tail or "") + "\t"
    for el in tree.xpath("//tr"):
        el.tail = (el.tail or "") + "\n"

    # Add line breaks after headings to preserve SEC section markers
    for el in tree.xpath("//h1|//h2|//h3|//h4|//h5|//h6"):
        el.tail = (el.tail or "") + "\n\n"

    text = tree.text_content()
    return _normalize_ws(text)


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


def _clean_inline_text(s: str) -> str:
    s = _INVISIBLES_RE.sub("", s or "")
    s = _UNISPACES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _candidate_caption(text: str, max_len: int = 300) -> bool:
    if not text or _CAPTION_DENY_RE.search(text):
        return False
    # A “caption-like” line is short and either:
    # - has table-ish keywords, or
    # - ends with a colon, or
    # - includes unit hints ( (in millions), (unaudited), etc. )
    if len(text) <= max_len and (
        _CAPTION_KEYS_RE.search(text) or text.endswith(":") or _CAPTION_UNITS_RE.search(text)
    ):
        return True
    return False


def _infer_table_caption(el, max_up: int = 2, max_back: int = 12, max_length: int = 1000) -> str:
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
            if _candidate_caption(text):
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
                if _candidate_caption(text):
                    return text[:max_length]
        node = node.getparent()
        if node is None:
            break

    return ""


def extract_tables_markdown_and_facts(html_bytes: bytes) -> list[dict[str, Any]]:
    """
    Returns a list of dicts:
    {
        'caption': str,
        'markdown': str,   # GitHub-style table
        'facts': [str],    # row-wise sentences
        'n_rows': int,
        'n_cols': int
    }
    """
    tree = fromstring(html_bytes)
    tables = tree.xpath("//table")
    out = []

    for t in tables:
        frag_html = tostring(t, encoding="unicode")
        try:
            dfs = pd.read_html(StringIO(frag_html), flavor="lxml")
        except Exception:
            # fallback: try html5lib if available
            try:
                dfs = pd.read_html(StringIO(frag_html), flavor="bs4")
            except Exception:
                continue  # skip malformed table

        if not dfs:
            continue

        # Some SEC tables parse into multiple frames; concatenate vertically when shapes align
        df = pd.concat(dfs, ignore_index=True, sort=False)

        df = _flatten_multiindex_columns(df)
        # Clean cells
        df = df.map(_clean_cell)
        # Drop empties
        df = _drop_empty_rows_cols(df)
        # Merge $/% unit columns
        df = _merge_unit_columns(df)
        if df.empty or df.shape[1] < 2:
            continue

        # Identify row label column (heuristic: first non-numeric-heavy column)
        row_label_col = df.columns[0]

        # Numeric normalization (but keep the row label as text)
        for c in df.columns:
            if c == row_label_col:
                continue
            df[c] = _normalize_numeric_series(df[c])

        # final drop of empty cols/rows after normalization
        df = _drop_empty_rows_cols(df)
        if df.empty or df.shape[1] < 2:
            continue

        # Produce Markdown
        md = df.to_markdown(index=False)
        
        # Using improved parsing method
        caption = _infer_table_caption(t)

        # Build facts: “<caption>: <row> — <col> = <value>”
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
                fact = f"{caption}: {row_label} — {col} = {val}" if caption else f"{row_label} — {col} = {val}"
                facts.append(fact)

        out.append(
            # TODO: use better dataclass
            {
                "caption": caption,
                "markdown": md,
                "facts": facts,
                "df": df,  # for debugging
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
            }
        )
    return out


def fetch_10ks_for_tickers(tickers: list[str], output_dir: Path, per_company: int = 2, delay: float = 0.2):
    out_raw_folder = output_dir / "10k_raw"
    out_meta_folder = output_dir / "meta"
    out_raw_folder.mkdir(parents=True, exist_ok=True)
    out_meta_folder.mkdir(parents=True, exist_ok=True)

    t2c = ticker_map()
    for t in tqdm(tickers, desc="fetching tickers"):
        cik = t2c[t.upper()]
        for acc_no, primary, fdate in tqdm(list_10k_submissions(cik, per_company), desc=f"processing ticker {t}"):
            html, src = download_primary(cik, acc_no, primary)

            text = extract_text_lxml_preserve_tables(html)
            # keep at most one blank line
            text = clean_invisible_and_blank_lines(text, keep_blank=1)

            # TODO:
            # - how to save these tables?
            # - should we remove tables from the above `text` to deduplicate information?
            tables = extract_tables_markdown_and_facts(html.encode("utf-8"))

            base = f"{t.upper()}_{acc_no}"

            (out_raw_folder / f"{base}.html").write_text(html, encoding="utf-8")
            (out_raw_folder / f"{base}.txt").write_text(text, encoding="utf-8")

            meta = {
                "ticker": t.upper(),
                "cik": cik,
                "filing_date": fdate,
                "accession": acc_no,
                "primary": primary,
                "source_url": src,
                "form": "10-K",
            }
            (out_meta_folder / f"{base}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

            time.sleep(delay)  # be polite

    logger.success("Done.")


if __name__ == "__main__":
    # TODO: get list of tickers from somewhere, e.g. top 500 companies
    # tickers = ["APH", "GOOGL", "NVDA"]
    tickers = ["APH"]
    out_dir = Path("./data")
    fetch_10ks_for_tickers(tickers, out_dir, per_company=1, delay=0.2)
