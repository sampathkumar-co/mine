from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
from pathlib import PurePosixPath
import re
import unicodedata
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit, unquote


ALLOWED_HOSTS = frozenset({
    "timeseriesclassification.com",
    "www.timeseriesclassification.com",
})
ROOT_URL = "https://www.timeseriesclassification.com/"
MAX_CRAWL_DEPTH = 2
MAX_HTML_PAGES = 500
ASCII_EDGE_WHITESPACE = "\t\n\v\f\r "
TRAVERSAL_SUFFIXES = frozenset({"", ".php", ".html", ".htm"})
DATA_SUFFIX_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:_|-)(?P<kind>train|test)"
    r"(?P<suffix>\.(?:ts|tsv|txt|csv|arff)(?:\.gz)?)$",
    re.IGNORECASE,
)
TRAIN_ANCHOR_TEXTS = frozenset({"train", "training data", "train file"})
TEST_ANCHOR_TEXTS = frozenset({"test", "testing data", "test file"})
FIELD_ALIASES = {
    "train_size": frozenset({
        "train size", "training size", "train instances", "training instances",
        "number of train cases", "number of training cases",
    }),
    "test_size": frozenset({
        "test size", "testing size", "test instances", "testing instances",
        "number of test cases", "number of testing cases",
    }),
    "series_length": frozenset({"series length", "time series length", "length"}),
    "class_count": frozenset({"class count", "classes", "number of classes", "number classes"}),
    "archive_type": frozenset({"archive type", "data type", "dimension", "dimensions", "type"}),
}
UNIVARIATE_VALUES = frozenset({
    "univariate",
    "univariate time series",
    "univariate time-series",
})
PROVENANCE_PATTERNS = {
    "UCI": (
        re.compile(r"\bUCI\b", re.IGNORECASE),
        re.compile(r"archive\.ics\.uci\.edu", re.IGNORECASE),
    ),
    "OpenML": (
        re.compile(r"\bOpenML\b", re.IGNORECASE),
        re.compile(r"openml\.org", re.IGNORECASE),
    ),
    "PMLB": (
        re.compile(r"\bPMLB\b", re.IGNORECASE),
        re.compile(r"EpistasisLab/pmlb", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class ParsedHTML:
    links: tuple[tuple[str, str], ...]
    rows: tuple[tuple[str, ...], ...]
    definitions: tuple[tuple[str, str], ...]
    text_nodes: tuple[str, ...]


@dataclass(frozen=True)
class CandidateMetadata:
    normalized_dataset_name: str
    total_instances: int
    series_length: int
    class_count: int
    classification: bool
    univariate: bool
    train_url: str
    test_url: str
    description_url: str
    metadata_page_sha256: str

    def canonical_fields(self) -> dict[str, object]:
        return {
            "normalized_dataset_name": self.normalized_dataset_name,
            "total_instances": self.total_instances,
            "series_length": self.series_length,
            "class_count": self.class_count,
            "classification": self.classification,
            "univariate": self.univariate,
            "train_url": self.train_url,
            "test_url": self.test_url,
        }


@dataclass(frozen=True)
class CandidateParseResult:
    description_url: str
    metadata: CandidateMetadata | None
    rejections: tuple[str, ...]


class _ArchiveHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.rows: list[tuple[str, ...]] = []
        self.definitions: list[tuple[str, str]] = []
        self.text_nodes: list[str] = []
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._in_row = False
        self._row: list[str] = []
        self._cell_tag: str | None = None
        self._cell_text: list[str] = []
        self._definition_tag: str | None = None
        self._definition_text: list[str] = []
        self._pending_dt: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag == "a":
            values = {key.casefold(): value for key, value in attrs}
            self._anchor_href = values.get("href")
            self._anchor_text = []
        elif tag == "tr":
            self._in_row = True
            self._row = []
        elif tag in {"td", "th"} and self._in_row:
            self._cell_tag = tag
            self._cell_text = []
        elif tag in {"dt", "dd"}:
            self._definition_tag = tag
            self._definition_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "a" and self._anchor_href is not None:
            self.links.append((self._anchor_href, normalize_visible_text(" ".join(self._anchor_text))))
            self._anchor_href = None
            self._anchor_text = []
        elif tag in {"td", "th"} and self._cell_tag == tag:
            self._row.append(normalize_visible_text(" ".join(self._cell_text)))
            self._cell_tag = None
            self._cell_text = []
        elif tag == "tr" and self._in_row:
            if self._cell_tag is not None:
                self._row.append(normalize_visible_text(" ".join(self._cell_text)))
            if any(self._row):
                self.rows.append(tuple(self._row))
            self._in_row = False
            self._row = []
            self._cell_tag = None
            self._cell_text = []
        elif tag in {"dt", "dd"} and self._definition_tag == tag:
            value = normalize_visible_text(" ".join(self._definition_text))
            if tag == "dt":
                self._pending_dt = value
            elif self._pending_dt is not None:
                self.definitions.append((self._pending_dt, value))
                self._pending_dt = None
            self._definition_tag = None
            self._definition_text = []

    def handle_data(self, data: str) -> None:
        value = normalize_visible_text(data)
        if not value:
            return
        self.text_nodes.append(value)
        if self._anchor_href is not None:
            self._anchor_text.append(value)
        if self._cell_tag is not None:
            self._cell_text.append(value)
        if self._definition_tag is not None:
            self._definition_text.append(value)


def normalize_visible_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_dataset_name(value: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFC", value).casefold()))


def normalize_field_label(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def decode_html(body: bytes) -> str:
    if body.startswith(b"\xef\xbb\xbf"):
        body = body[3:]
    return body.decode("utf-8", errors="strict")


def _uppercase_percent_triplets(value: str) -> str:
    return re.sub(r"%[0-9a-fA-F]{2}", lambda match: match.group(0).upper(), value)


def canonical_url(base_url: str, href: str) -> str:
    if not isinstance(href, str):
        raise TypeError("href must be str")
    raw = unicodedata.normalize("NFC", href.strip(ASCII_EDGE_WHITESPACE))
    joined = urljoin(base_url, raw)
    parts = urlsplit(joined)
    if parts.scheme.casefold() != "https":
        raise ValueError("URL must use HTTPS")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URL credentials are forbidden")
    host = (parts.hostname or "").encode("idna").decode("ascii").casefold()
    if host not in ALLOWED_HOSTS:
        raise ValueError("URL host is outside the frozen official hosts")
    if parts.port not in (None, 443):
        raise ValueError("non-default URL port is forbidden")
    path = quote(unicodedata.normalize("NFC", parts.path or "/"), safe="/%:@-._~!$&'()*+,;=")
    path = _uppercase_percent_triplets(path)
    pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True, encoding="utf-8", errors="strict")
    normalized_pairs = sorted(
        (unicodedata.normalize("NFC", key), unicodedata.normalize("NFC", value))
        for key, value in pairs
    )
    query = urlencode(normalized_pairs, doseq=True, safe="-._~", encoding="utf-8", errors="strict", quote_via=quote)
    netloc = host
    return urlunsplit(("https", netloc, path, query, ""))


def parse_html(body: bytes) -> ParsedHTML:
    parser = _ArchiveHTMLParser()
    parser.feed(decode_html(body))
    parser.close()
    return ParsedHTML(
        links=tuple(parser.links),
        rows=tuple(parser.rows),
        definitions=tuple(parser.definitions),
        text_nodes=tuple(parser.text_nodes),
    )


def candidate_name_from_url(url: str) -> str:
    canonical = canonical_url(ROOT_URL, url)
    parts = urlsplit(canonical)
    pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True, encoding="utf-8", errors="strict")
    if len(pairs) != 1 or pairs[0][0] != "Dataset" or not pairs[0][1]:
        raise ValueError("candidate URL must contain exactly one nonempty Dataset query")
    name = normalize_dataset_name(pairs[0][1].strip(ASCII_EDGE_WHITESPACE))
    if not name:
        raise ValueError("candidate dataset name normalizes to empty")
    return name


def classify_catalogue_link(url: str) -> str | None:
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True, encoding="utf-8", errors="strict")
    if parts.path.casefold().endswith("/description.php"):
        if len(pairs) == 1 and pairs[0][0] == "Dataset" and pairs[0][1]:
            return "candidate"
        return None
    if pairs:
        return None
    suffix = PurePosixPath(parts.path).suffix.casefold()
    if suffix in TRAVERSAL_SUFFIXES:
        return "traversal"
    return None


def catalogue_links(page_url: str, body: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    traversal: set[str] = set()
    candidates: set[str] = set()
    for href, _ in parse_html(body).links:
        try:
            url = canonical_url(page_url, href)
            kind = classify_catalogue_link(url)
        except (TypeError, ValueError, UnicodeError):
            continue
        if kind == "traversal":
            traversal.add(url)
        elif kind == "candidate":
            candidates.add(url)
    return tuple(sorted(traversal)), tuple(sorted(candidates))


def catalogue_plan(
    root_url: str,
    pages: dict[str, bytes],
    *,
    maximum_depth: int = MAX_CRAWL_DEPTH,
    maximum_pages: int = MAX_HTML_PAGES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = canonical_url(ROOT_URL, root_url)
    frontier = (root,)
    visited: set[str] = set()
    ordered: list[str] = []
    candidates: set[str] = set()
    for depth in range(maximum_depth + 1):
        next_frontier: set[str] = set()
        for url in sorted(frontier):
            if url in visited:
                continue
            if url not in pages:
                raise KeyError(f"scheduled catalogue page missing: {url}")
            visited.add(url)
            ordered.append(url)
            if len(ordered) > maximum_pages:
                raise OverflowError("catalogue page limit exceeded")
            traversal, found = catalogue_links(url, pages[url])
            candidates.update(found)
            if depth < maximum_depth:
                next_frontier.update(link for link in traversal if link not in visited)
        frontier = tuple(sorted(next_frontier))
    return tuple(ordered), tuple(sorted(candidates))


def _parse_positive_integer(value: str) -> int:
    raw = value.strip(ASCII_EDGE_WHITESPACE)
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
        digits = raw
    elif re.fullmatch(r"[1-9][0-9]{0,2}(?:,[0-9]{3})+", raw):
        digits = raw.replace(",", "")
    else:
        raise ValueError("metadata integer has noncanonical syntax")
    if len(digits) > 1 and digits.startswith("0"):
        raise ValueError("metadata integer has a leading zero")
    result = int(digits)
    if result <= 0 or result > 999_999_999:
        raise ValueError("metadata integer is outside the frozen range")
    return result


def _recognized_values(parsed: ParsedHTML) -> dict[str, list[str]]:
    output = {key: [] for key in FIELD_ALIASES}
    pairs: list[tuple[str, str]] = []
    for row in parsed.rows:
        if len(row) >= 2:
            pairs.append((row[0], " ".join(row[1:])))
    pairs.extend(parsed.definitions)
    for text in parsed.text_nodes:
        if text.count(":") == 1:
            pairs.append(tuple(part.strip() for part in text.split(":", 1)))
    alias_to_field = {
        alias: field
        for field, aliases in FIELD_ALIASES.items()
        for alias in aliases
    }
    for label, value in pairs:
        field = alias_to_field.get(normalize_field_label(label))
        normalized_value = normalize_visible_text(value)
        if field is not None and normalized_value:
            output[field].append(normalized_value)
    return output


def _one_distinct(
    values: list[str],
    field: str,
    converter,
) -> object:
    if not values:
        raise KeyError(f"missing:{field}")
    converted = {converter(value) for value in values}
    if len(converted) != 1:
        raise ValueError(f"conflict:{field}")
    return next(iter(converted))


def _data_link_kind(url: str, anchor_text: str, dataset_name: str) -> str | None:
    path_name = PurePosixPath(unquote(urlsplit(url).path)).name
    match = DATA_SUFFIX_PATTERN.fullmatch(path_name)
    if match is None:
        return None
    if normalize_dataset_name(match.group("prefix")) != dataset_name:
        return None
    kind = match.group("kind").casefold()
    text = normalize_field_label(anchor_text)
    if kind == "train" and (not text or text in TRAIN_ANCHOR_TEXTS):
        return "train"
    if kind == "test" and (not text or text in TEST_ANCHOR_TEXTS):
        return "test"
    if text in TRAIN_ANCHOR_TEXTS:
        return "train"
    if text in TEST_ANCHOR_TEXTS:
        return "test"
    return kind


def parse_candidate_page(description_url: str, body: bytes) -> CandidateParseResult:
    try:
        url = canonical_url(ROOT_URL, description_url)
        dataset_name = candidate_name_from_url(url)
        text = decode_html(body)
        parsed = parse_html(body)
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        return CandidateParseResult(description_url, None, (f"page:{error}",))

    rejections: list[str] = []
    for source, patterns in PROVENANCE_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            rejections.append(f"provenance:{source}")
    values = _recognized_values(parsed)
    extracted: dict[str, object] = {}
    for field in ("train_size", "test_size", "series_length", "class_count"):
        try:
            extracted[field] = _one_distinct(values[field], field, _parse_positive_integer)
        except (KeyError, ValueError) as error:
            rejections.append(str(error))
    try:
        archive_type = _one_distinct(values["archive_type"], "archive_type", normalize_field_label)
        if archive_type not in UNIVARIATE_VALUES:
            rejections.append("not-univariate")
    except (KeyError, ValueError) as error:
        rejections.append(str(error))

    links: dict[str, set[str]] = {"train": set(), "test": set()}
    for href, anchor_text in parsed.links:
        try:
            link = canonical_url(url, href)
        except (TypeError, ValueError, UnicodeError):
            continue
        kind = _data_link_kind(link, anchor_text, dataset_name)
        if kind is not None:
            links[kind].add(link)
    for kind in ("train", "test"):
        if not links[kind]:
            rejections.append(f"missing:{kind}_url")
        elif len(links[kind]) > 1:
            rejections.append(f"conflict:{kind}_url")

    if rejections:
        return CandidateParseResult(url, None, tuple(sorted(set(rejections))))
    metadata = CandidateMetadata(
        normalized_dataset_name=dataset_name,
        total_instances=int(extracted["train_size"]) + int(extracted["test_size"]),
        series_length=int(extracted["series_length"]),
        class_count=int(extracted["class_count"]),
        classification=True,
        univariate=True,
        train_url=next(iter(links["train"])),
        test_url=next(iter(links["test"])),
        description_url=url,
        metadata_page_sha256=hashlib.sha256(body).hexdigest(),
    )
    return CandidateParseResult(url, metadata, ())


def reject_duplicate_names(
    results: tuple[CandidateParseResult, ...],
) -> tuple[tuple[CandidateMetadata, ...], dict[str, tuple[str, ...]]]:
    groups: dict[str, list[CandidateMetadata]] = {}
    rejections: dict[str, tuple[str, ...]] = {}
    for result in results:
        if result.metadata is None:
            rejections[result.description_url] = result.rejections
            continue
        groups.setdefault(result.metadata.normalized_dataset_name, []).append(result.metadata)
    accepted: list[CandidateMetadata] = []
    for name, rows in sorted(groups.items()):
        distinct_urls = {row.description_url for row in rows}
        if len(distinct_urls) != 1 or len(rows) != 1:
            for row in rows:
                rejections[row.description_url] = ("duplicate-normalized-name",)
            continue
        accepted.append(rows[0])
    return tuple(sorted(accepted, key=lambda row: (row.normalized_dataset_name, row.description_url))), rejections


def source_release_manifest(
    pages: tuple[tuple[str, str, bytes], ...],
) -> tuple[bytes, str]:
    rows: list[list[str]] = []
    seen: set[str] = set()
    for requested_url, final_url, body in pages:
        requested = canonical_url(ROOT_URL, requested_url)
        final = canonical_url(requested, final_url)
        if requested in seen:
            raise ValueError("duplicate requested catalogue URL")
        seen.add(requested)
        rows.append([requested, final, hashlib.sha256(body).hexdigest()])
    rows.sort(key=lambda row: row[0])
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return encoded, hashlib.sha256(encoded).hexdigest()
