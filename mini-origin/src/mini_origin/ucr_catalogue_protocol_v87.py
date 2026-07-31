from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import math
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
NON_VISIBLE_TEXT_TAGS = frozenset({
    "canvas", "head", "noscript", "script", "style", "svg", "template",
})
HTML_PAGE_ATTEMPT_COUNT = 3
HTML_PAGE_ATTEMPT_INDICES = (1, 2, 3)
HTML_PAGE_RETRY_DELAYS_SECONDS = (0, 5, 20)
HTML_PAGE_REQUEST_METHOD = "GET"
HTML_PAGE_REQUEST_HEADERS = (
    ("Accept", "text/html,application/xhtml+xml,application/json,text/csv;q=0.9,*/*;q=0.1"),
    ("Accept-Encoding", "identity"),
    ("Cache-Control", "no-cache"),
    ("User-Agent", "Mini-ORIGIN-v0.87-UCR-byte-lock/1.0"),
)
HTML_PAGE_CONNECT_TIMEOUT_SECONDS = 15.0
HTML_PAGE_READ_TIMEOUT_SECONDS = 60.0
HTML_PAGE_TOTAL_DEADLINE_SECONDS = 90.0
HTML_PAGE_MAX_REDIRECTS = 5
HTML_PAGE_MAX_RESPONSE_BYTES = 10_000_000
MIN_TOTAL_INSTANCES = 300
MAX_TOTAL_INSTANCES = 6_000
MIN_SERIES_LENGTH = 4
MAX_SERIES_LENGTH = 80
MIN_CLASS_COUNT = 2
MAX_CLASS_COUNT = 15
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


@dataclass(frozen=True)
class HTMLPageAttempt:
    attempt_index: int
    scheduled_delay_seconds: int
    request_method: str
    request_headers: tuple[tuple[str, str], ...]
    tls_certificate_validated: bool
    redirect_chain: tuple[str, ...]
    connect_elapsed_seconds: tuple[float, ...]
    read_elapsed_seconds: float | None
    total_elapsed_seconds: float | None
    status_code: int | None
    final_url: str | None
    body: bytes | None
    failure: str | None = None


@dataclass(frozen=True)
class AuthoritativeHTMLPage:
    requested_url: str
    final_url: str
    body: bytes
    body_sha256: str
    authoritative_attempt_index: int
    attempts: tuple[HTMLPageAttempt, ...]


@dataclass(frozen=True)
class MetadataSnapshot:
    crawl_page_urls: tuple[str, ...]
    candidate_urls: tuple[str, ...]
    source_manifest_bytes: bytes
    frozen_source_release: str
    eligible_metadata: tuple[CandidateMetadata, ...]
    rejections: tuple[tuple[str, tuple[str, ...]], ...]


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
        self._hidden_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if self._hidden_stack:
            if tag in NON_VISIBLE_TEXT_TAGS:
                self._hidden_stack.append(tag)
            return
        if tag in NON_VISIBLE_TEXT_TAGS:
            self._hidden_stack.append(tag)
            return
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
        if self._hidden_stack:
            if tag == self._hidden_stack[-1]:
                self._hidden_stack.pop()
            return
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
        if self._hidden_stack:
            return
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
    pieces: list[str] = []
    separator_pending = False
    for character in normalized:
        if character.isalnum():
            if separator_pending and pieces:
                pieces.append(" ")
            pieces.append(character)
            separator_pending = False
        else:
            separator_pending = True
    return "".join(pieces).strip()


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
    raw_path = parts.path or "/"
    if re.search(r"%(?![0-9a-fA-F]{2})", raw_path):
        raise ValueError("URL path contains a malformed percent triplet")
    decoded_path = unquote(raw_path, encoding="utf-8", errors="strict")
    path = quote(
        unicodedata.normalize("NFC", decoded_path),
        safe="/:@-._~!$&'()*+,;=",
    )
    path = _uppercase_percent_triplets(path)
    pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True, encoding="utf-8", errors="strict")
    normalized_pairs = sorted(
        (unicodedata.normalize("NFC", key), unicodedata.normalize("NFC", value))
        for key, value in pairs
    )
    query = urlencode(normalized_pairs, doseq=True, safe="-._~", encoding="utf-8", errors="strict", quote_via=quote)
    netloc = host
    return urlunsplit(("https", netloc, path, query, ""))


def _finite_nonnegative(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def authoritative_html_page(
    requested_url: str,
    attempts: tuple[HTMLPageAttempt, ...],
) -> AuthoritativeHTMLPage:
    requested = canonical_url(ROOT_URL, requested_url)
    if len(attempts) != HTML_PAGE_ATTEMPT_COUNT:
        raise ValueError("every HTML page requires exactly three attempt records")
    indices = tuple(sorted(attempt.attempt_index for attempt in attempts))
    if indices != HTML_PAGE_ATTEMPT_INDICES:
        raise ValueError("HTML page attempt indices must be exactly 1, 2 and 3")

    successes: list[tuple[int, str, bytes, str]] = []
    for attempt in attempts:
        expected_delay = HTML_PAGE_RETRY_DELAYS_SECONDS[attempt.attempt_index - 1]
        if attempt.scheduled_delay_seconds != expected_delay:
            raise ValueError("HTML page retry delay differs from frozen schedule")
        if attempt.request_method != HTML_PAGE_REQUEST_METHOD:
            raise ValueError("HTML page request method differs from frozen method")
        if tuple(attempt.request_headers) != HTML_PAGE_REQUEST_HEADERS:
            raise ValueError("HTML page request headers differ from frozen headers")
        if attempt.failure is not None:
            if attempt.body is not None:
                raise ValueError("failed HTML page attempts cannot contain accepted body bytes")
            if attempt.final_url is not None or attempt.status_code is not None:
                raise ValueError("failed HTML page attempts cannot contain success-state fields")
            if (
                attempt.tls_certificate_validated is not False
                or attempt.redirect_chain
                or attempt.connect_elapsed_seconds
                or attempt.read_elapsed_seconds is not None
                or attempt.total_elapsed_seconds is not None
            ):
                raise ValueError(
                    "failed HTML page attempts cannot contain transport-success evidence"
                )
            normalized_failure = normalize_visible_text(attempt.failure)
            if not normalized_failure or normalized_failure != attempt.failure:
                raise ValueError(
                    "failed HTML page attempts require normalized non-empty failure text"
                )
            continue
        if attempt.final_url is None or attempt.body is None:
            raise ValueError("successful HTML page attempts require final URL and body")
        if attempt.tls_certificate_validated is not True:
            raise ValueError("successful HTML page attempts require validated TLS")
        if attempt.status_code != 200:
            raise ValueError("successful HTML page attempts require HTTP 200")
        if len(attempt.redirect_chain) > HTML_PAGE_MAX_REDIRECTS:
            raise ValueError("HTML page redirect limit exceeded")
        canonical_redirects = tuple(
            canonical_url(requested, redirect_url)
            for redirect_url in attempt.redirect_chain
        )
        final = canonical_url(requested, attempt.final_url)
        expected_final = canonical_redirects[-1] if canonical_redirects else requested
        if final != expected_final:
            raise ValueError(
                "HTML page final URL differs from the frozen redirect-chain endpoint"
            )
        expected_connections = len(attempt.redirect_chain) + 1
        if len(attempt.connect_elapsed_seconds) != expected_connections:
            raise ValueError("HTML page connection timing count differs from redirect chain")
        if any(
            _finite_nonnegative(value, "connect elapsed seconds")
            > HTML_PAGE_CONNECT_TIMEOUT_SECONDS
            for value in attempt.connect_elapsed_seconds
        ):
            raise ValueError("HTML page connect timeout exceeded")
        if attempt.read_elapsed_seconds is None or attempt.total_elapsed_seconds is None:
            raise ValueError("successful HTML page attempts require read and total timings")
        if (
            _finite_nonnegative(attempt.read_elapsed_seconds, "read elapsed seconds")
            > HTML_PAGE_READ_TIMEOUT_SECONDS
        ):
            raise ValueError("HTML page read timeout exceeded")
        if (
            _finite_nonnegative(attempt.total_elapsed_seconds, "total elapsed seconds")
            > HTML_PAGE_TOTAL_DEADLINE_SECONDS
        ):
            raise ValueError("HTML page total deadline exceeded")
        if not isinstance(attempt.body, bytes):
            raise TypeError("HTML page body must be bytes")
        if not 1 <= len(attempt.body) <= HTML_PAGE_MAX_RESPONSE_BYTES:
            raise ValueError("HTML page body is empty or exceeds the frozen byte cap")
        digest = hashlib.sha256(attempt.body).hexdigest()
        successes.append((attempt.attempt_index, final, attempt.body, digest))

    if not successes:
        raise RuntimeError("all HTML page attempts failed")
    final_urls = {row[1] for row in successes}
    bodies = {row[2] for row in successes}
    digests = {row[3] for row in successes}
    if len(final_urls) != 1 or len(bodies) != 1 or len(digests) != 1:
        raise RuntimeError("successful HTML page attempts disagree")
    index, final, body, digest = min(successes, key=lambda row: row[0])
    return AuthoritativeHTMLPage(
        requested_url=requested,
        final_url=final,
        body=body,
        body_sha256=digest,
        authoritative_attempt_index=index,
        attempts=tuple(sorted(attempts, key=lambda attempt: attempt.attempt_index)),
    )


def revalidate_authoritative_html_page(
    page: AuthoritativeHTMLPage,
) -> AuthoritativeHTMLPage:
    validated = authoritative_html_page(page.requested_url, page.attempts)
    if validated != page:
        raise ValueError("authoritative HTML page differs from retained attempt evidence")
    return validated


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
    pages: dict[str, AuthoritativeHTMLPage],
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
            page = revalidate_authoritative_html_page(pages[url])
            if canonical_url(ROOT_URL, page.requested_url) != url:
                raise ValueError("catalogue page mapping key differs from requested URL")
            visited.add(url)
            ordered.append(url)
            if len(ordered) > maximum_pages:
                raise OverflowError("catalogue page limit exceeded")
            traversal, found = catalogue_links(page.final_url, page.body)
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


def _has_data_suffix(path_name: str) -> bool:
    return bool(re.search(r"\.(?:ts|tsv|txt|csv|arff)(?:\.gz)?$", path_name, re.IGNORECASE))


def _data_link_kind(url: str, anchor_text: str, dataset_name: str) -> str | None:
    path_name = PurePosixPath(unquote(urlsplit(url).path)).name
    text = normalize_field_label(anchor_text)
    match = DATA_SUFFIX_PATTERN.fullmatch(path_name)
    if match is not None and normalize_dataset_name(match.group("prefix")) == dataset_name:
        return match.group("kind").casefold()
    if not _has_data_suffix(path_name):
        return None
    if text in TRAIN_ANCHOR_TEXTS:
        return "train"
    if text in TEST_ANCHOR_TEXTS:
        return "test"
    return None


def parse_candidate_page(
    description_url: str,
    body: bytes,
    *,
    link_base_url: str | None = None,
) -> CandidateParseResult:
    try:
        url = canonical_url(ROOT_URL, description_url)
        link_base = canonical_url(url, link_base_url or url)
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
            link = canonical_url(link_base, href)
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
    if (
        len(links["train"]) == 1
        and len(links["test"]) == 1
        and next(iter(links["train"])) == next(iter(links["test"]))
    ):
        rejections.append("train-test-url-identical")

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


def validate_candidate_result_coverage(
    discovered_candidate_urls: tuple[str, ...],
    results: tuple[CandidateParseResult, ...],
) -> None:
    discovered = tuple(
        sorted(canonical_url(ROOT_URL, url) for url in discovered_candidate_urls)
    )
    if len(set(discovered)) != len(discovered):
        raise ValueError("discovered candidate URLs contain duplicates")
    result_urls = tuple(
        canonical_url(ROOT_URL, result.description_url) for result in results
    )
    if len(set(result_urls)) != len(result_urls):
        raise ValueError("candidate parse results contain duplicate URLs")
    missing = sorted(set(discovered) - set(result_urls))
    extra = sorted(set(result_urls) - set(discovered))
    if missing or extra:
        raise RuntimeError(
            f"candidate result coverage mismatch: missing={missing!r} extra={extra!r}"
        )
    require_no_candidate_page_failures(results)


def parse_authoritative_candidate_page(
    page: AuthoritativeHTMLPage,
) -> CandidateParseResult:
    page = revalidate_authoritative_html_page(page)
    candidate_name_from_url(page.requested_url)
    requested_parts = urlsplit(page.requested_url)
    final_parts = urlsplit(page.final_url)
    if (requested_parts.path, requested_parts.query) != (
        final_parts.path,
        final_parts.query,
    ):
        raise RuntimeError("candidate redirect changed description path or identity")
    return parse_candidate_page(
        page.requested_url,
        page.body,
        link_base_url=page.final_url,
    )


def require_no_candidate_page_failures(
    results: tuple[CandidateParseResult, ...],
) -> None:
    failures = [
        (result.description_url, rejection)
        for result in results
        for rejection in result.rejections
        if rejection.startswith("page:")
    ]
    if failures:
        raise RuntimeError(f"candidate page failure: {failures!r}")


def metadata_numeric_series_eligible(metadata: CandidateMetadata) -> bool:
    try:
        train_url = canonical_url(metadata.description_url, metadata.train_url)
        test_url = canonical_url(metadata.description_url, metadata.test_url)
    except (TypeError, ValueError, UnicodeError):
        return False
    if train_url != metadata.train_url or test_url != metadata.test_url:
        return False
    train_name = PurePosixPath(unquote(urlsplit(train_url).path)).name
    test_name = PurePosixPath(unquote(urlsplit(test_url).path)).name
    return (
        metadata.classification is True
        and metadata.univariate is True
        and train_url != test_url
        and _has_data_suffix(train_name)
        and _has_data_suffix(test_name)
    )


def metadata_eligibility_rejections(
    metadata: CandidateMetadata,
) -> tuple[str, ...]:
    rejections: list[str] = []
    if metadata.classification is not True:
        rejections.append("not-classification")
    if metadata.univariate is not True:
        rejections.append("not-univariate")
    if not MIN_TOTAL_INSTANCES <= metadata.total_instances <= MAX_TOTAL_INSTANCES:
        rejections.append("total-instances-outside-range")
    if not MIN_SERIES_LENGTH <= metadata.series_length <= MAX_SERIES_LENGTH:
        rejections.append("series-length-outside-range")
    if not MIN_CLASS_COUNT <= metadata.class_count <= MAX_CLASS_COUNT:
        rejections.append("class-count-outside-range")
    if not metadata_numeric_series_eligible(metadata):
        rejections.append("numeric-series-metadata-predicate-failed")
    return tuple(sorted(rejections))


def reject_duplicate_names(
    results: tuple[CandidateParseResult, ...],
) -> tuple[tuple[CandidateMetadata, ...], dict[str, tuple[str, ...]]]:
    groups: dict[str, list[CandidateParseResult]] = {}
    for result in results:
        name = candidate_name_from_url(result.description_url)
        groups.setdefault(name, []).append(result)
    accepted: list[CandidateMetadata] = []
    rejections: dict[str, tuple[str, ...]] = {}
    for name, rows in sorted(groups.items()):
        distinct_urls = {canonical_url(ROOT_URL, row.description_url) for row in rows}
        duplicate = len(distinct_urls) != 1 or len(rows) != 1
        for row in rows:
            url = canonical_url(ROOT_URL, row.description_url)
            reasons = set(row.rejections)
            if duplicate:
                reasons.add("duplicate-normalized-name")
            if reasons:
                rejections[url] = tuple(sorted(reasons))
            elif row.metadata is not None:
                accepted.append(row.metadata)
            else:
                raise RuntimeError("candidate result has neither metadata nor rejection")
    return tuple(sorted(
        accepted,
        key=lambda row: (row.normalized_dataset_name, row.description_url),
    )), rejections


def source_release_manifest(
    crawl_page_urls: tuple[str, ...],
    candidate_urls: tuple[str, ...],
    pages: dict[str, AuthoritativeHTMLPage],
) -> tuple[bytes, str]:
    expected = tuple(sorted(
        {canonical_url(ROOT_URL, url) for url in crawl_page_urls}
        | {canonical_url(ROOT_URL, url) for url in candidate_urls}
    ))
    canonical_pages: dict[str, AuthoritativeHTMLPage] = {}
    for key, raw_page in pages.items():
        canonical_key = canonical_url(ROOT_URL, key)
        page = revalidate_authoritative_html_page(raw_page)
        if canonical_url(ROOT_URL, page.requested_url) != canonical_key:
            raise ValueError("source manifest page key differs from requested URL")
        if canonical_key in canonical_pages:
            raise ValueError("duplicate requested catalogue URL")
        canonical_pages[canonical_key] = page
    missing = sorted(set(expected) - set(canonical_pages))
    extra = sorted(set(canonical_pages) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"source manifest coverage mismatch: missing={missing!r} extra={extra!r}"
        )
    rows = [
        [url, canonical_pages[url].final_url, canonical_pages[url].body_sha256]
        for url in expected
    ]
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return encoded, hashlib.sha256(encoded).hexdigest()


def finalize_metadata_snapshot(
    root_url: str,
    pages: dict[str, AuthoritativeHTMLPage],
    excluded_normalized_names: frozenset[str],
) -> MetadataSnapshot:
    normalized_exclusions = frozenset(
        normalize_dataset_name(name) for name in excluded_normalized_names
    )
    if normalized_exclusions != excluded_normalized_names:
        raise ValueError("v0.86 exclusion registry names must already be normalized")
    crawl_page_urls, candidate_urls = catalogue_plan(root_url, pages)
    if len(set(crawl_page_urls) | set(candidate_urls)) > MAX_HTML_PAGES:
        raise OverflowError("complete catalogue HTML page limit exceeded")
    results: list[CandidateParseResult] = []
    for candidate_url in candidate_urls:
        if candidate_url not in pages:
            raise KeyError(f"discovered candidate page missing: {candidate_url}")
        results.append(parse_authoritative_candidate_page(pages[candidate_url]))
    result_tuple = tuple(results)
    validate_candidate_result_coverage(candidate_urls, result_tuple)
    manifest_bytes, frozen_source_release = source_release_manifest(
        crawl_page_urls, candidate_urls, pages
    )
    accepted, rejected = reject_duplicate_names(result_tuple)
    eligible: list[CandidateMetadata] = []
    rejection_map = dict(rejected)
    for metadata in accepted:
        row_rejections: list[str] = []
        if metadata.normalized_dataset_name in normalized_exclusions:
            row_rejections.append("name-in-v86-contamination-registry")
        row_rejections.extend(metadata_eligibility_rejections(metadata))
        if row_rejections:
            rejection_map[metadata.description_url] = tuple(sorted(set(row_rejections)))
        else:
            eligible.append(metadata)
    return MetadataSnapshot(
        crawl_page_urls=crawl_page_urls,
        candidate_urls=candidate_urls,
        source_manifest_bytes=manifest_bytes,
        frozen_source_release=frozen_source_release,
        eligible_metadata=tuple(sorted(
            eligible,
            key=lambda row: (row.normalized_dataset_name, row.description_url),
        )),
        rejections=tuple(sorted(rejection_map.items())),
    )
