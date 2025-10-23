import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from loguru import logger
from lxml.html import HtmlMixin, fromstring
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


def _normalize_ws(text: str) -> str:
    # collapse spaces/tabs, normalize newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u00A0", " ", text)  # &nbsp;
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_isolated_page_number(s: str) -> bool:
    # e.g., "99", "102", often a single line page header/footer number
    return bool(re.fullmatch(r"\d{1,4}", s))


def _clean_inline_text(s: HtmlMixin | str) -> str:
    # NEW: accept lxml elements too
    if isinstance(s, HtmlMixin):
        s = s.text_content()
    s = s or ""
    s = _INVISIBLES_RE.sub("", s)
    s = _UNISPACES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def extract_table(table_el: HtmlMixin) -> str:
    rows = []
    for tr in table_el.xpath(".//tr"):
        cells = [c.text_content().strip().replace("\n", " ") for c in tr.xpath(".//th|.//td")]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if rows:
        num_cols = rows[0].split("|")[1:-1]
        sep = "| " + " | ".join(["---"] * len(num_cols)) + " |"
        return "\n".join([rows[0], sep] + rows[1:]) + "\n\n"
    return ""


# TODO: fix lxml typing
def extract_text_with_tables(html: str) -> str:
    tree = fromstring(html.encode("utf-8"))
    _drop_toc_headers(tree)

    # identifies tables and replace them with markdown representation in-line
    for table_el in tree.xpath("//table"):
        table_md = extract_table(table_el)
        if table_md:
            table_placeholder = tree.makeelement("div")
            table_placeholder.text = f"\n\n{table_md}\n\n"
            table_el.addnext(table_placeholder)
            parent = table_el.getparent()
            if parent is not None:
                parent.remove(table_el)

    # linearize
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


#################################
# MAIN, PUBLIC FUNCTION
#################################
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

            text = extract_text_with_tables(html)

            # write outputs
            norm_dir = out_dir / "normalized" / base
            norm_dir.mkdir(parents=True, exist_ok=True)

            # write 10K metadata
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

            ########################
            # write markdown file for easier sanity checking
            ########################
            md_dir = norm_dir / "markdown"
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / "content.md").write_text(text, encoding="utf-8")

            ########################
            # write raw data
            ########################
            (out_raw_folder / f"{base}.html").write_text(html, encoding="utf-8")
            (out_raw_folder / f"{base}.txt").write_text(text, encoding="utf-8")

            # be polite to SEC servers
            time.sleep(delay)

    logger.success("Done.")


if __name__ == "__main__":
    # TODO: get list of tickers from somewhere, e.g. top 500 companies
    tickers = ["APH", "GOOGL", "NVDA", "AMZN", "MSFT", "META", "TSLA", "AAPL", "TSM"]
    out_dir = Path("./data")
    fetch_10ks_for_tickers(tickers, out_dir, per_company=5, delay=0.1)
