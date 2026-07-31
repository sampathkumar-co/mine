from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mini_origin import ucr_catalogue_protocol_v87 as catalogue


def candidate_html(*, conflict: bool = False, provenance: str = "") -> bytes:
    conflict_row = "<tr><th>Train size</th><td>201</td></tr>" if conflict else ""
    return f"""<!doctype html>
    <html><body>{provenance}
      <table>
        <tr><th>Train size</th><td>200</td></tr>
        {conflict_row}
        <tr><th>Test size</th><td>121</td></tr>
        <tr><th>Time series length</th><td>24</td></tr>
        <tr><th>Number of classes</th><td>3</td></tr>
        <tr><th>Type</th><td>Univariate</td></tr>
      </table>
      <a href="/data/FixtureSeries_TRAIN.tsv">TRAIN</a>
      <a href="/data/FixtureSeries_TEST.tsv">TEST</a>
    </body></html>""".encode("utf-8")


def test_canonical_url_is_same_host_https_and_query_sorted():
    value = catalogue.canonical_url(
        catalogue.ROOT_URL,
        " /description.php?z=2&Dataset=Caf\u00e9 ",
    )
    assert value == (
        "https://www.timeseriesclassification.com/description.php?"
        "Dataset=Caf%C3%A9&z=2"
    )
    with pytest.raises(ValueError):
        catalogue.canonical_url(catalogue.ROOT_URL, "http://example.com/data")


def test_html_decoding_is_strict_with_optional_utf8_bom():
    assert catalogue.decode_html(b"\xef\xbb\xbfhello") == "hello"
    with pytest.raises(UnicodeDecodeError):
        catalogue.decode_html(b"\xff")


def test_catalogue_plan_is_breadth_first_lexicographic_and_deterministic():
    root = catalogue.ROOT_URL
    listing_a = catalogue.canonical_url(root, "/archive.html")
    listing_b = catalogue.canonical_url(root, "/dataset.php")
    candidate_a = catalogue.canonical_url(root, "/description.php?Dataset=Beta")
    candidate_b = catalogue.canonical_url(root, "/description.php?Dataset=Alpha")
    pages = {
        root: (
            '<a href="/dataset.php">datasets</a>'
            '<a href="/archive.html">archive</a>'
        ).encode(),
        listing_a: f'<a href="{candidate_a}">Beta</a>'.encode(),
        listing_b: f'<a href="{candidate_b}">Alpha</a>'.encode(),
    }
    order, candidates = catalogue.catalogue_plan(root, pages)
    assert order == (root, listing_a, listing_b)
    assert candidates == tuple(sorted((candidate_a, candidate_b)))
    reordered = dict(reversed(tuple(pages.items())))
    assert catalogue.catalogue_plan(root, reordered) == (order, candidates)


def test_candidate_parser_extracts_exact_required_metadata():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL,
        "/description.php?Dataset=FixtureSeries",
    )
    result = catalogue.parse_candidate_page(url, candidate_html())
    assert result.rejections == ()
    assert result.metadata is not None
    assert result.metadata.canonical_fields() == {
        "normalized_dataset_name": "fixtureseries",
        "total_instances": 321,
        "series_length": 24,
        "class_count": 3,
        "classification": True,
        "univariate": True,
        "train_url": "https://www.timeseriesclassification.com/data/FixtureSeries_TRAIN.tsv",
        "test_url": "https://www.timeseriesclassification.com/data/FixtureSeries_TEST.tsv",
    }
    assert result.metadata.metadata_page_sha256 == hashlib.sha256(candidate_html()).hexdigest()


def test_conflicting_recognized_values_reject_without_precedence():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL,
        "/description.php?Dataset=FixtureSeries",
    )
    result = catalogue.parse_candidate_page(url, candidate_html(conflict=True))
    assert result.metadata is None
    assert "conflict:train_size" in result.rejections


def test_missing_required_metadata_rejects_candidate():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL,
        "/description.php?Dataset=FixtureSeries",
    )
    body = candidate_html().replace(b"<tr><th>Test size</th><td>121</td></tr>", b"")
    result = catalogue.parse_candidate_page(url, body)
    assert result.metadata is None
    assert "'missing:test_size'" in result.rejections


def test_provenance_match_rejects_before_selection():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL,
        "/description.php?Dataset=FixtureSeries",
    )
    result = catalogue.parse_candidate_page(
        url,
        candidate_html(provenance="Source: OpenML"),
    )
    assert result.metadata is None
    assert "provenance:OpenML" in result.rejections


def test_duplicate_normalized_names_reject_every_row():
    one = catalogue.parse_candidate_page(
        catalogue.canonical_url(catalogue.ROOT_URL, "/description.php?Dataset=FixtureSeries"),
        candidate_html(),
    )
    two = catalogue.parse_candidate_page(
        catalogue.canonical_url(catalogue.ROOT_URL, "/other/description.php?Dataset=FixtureSeries"),
        candidate_html(),
    )
    accepted, rejected = catalogue.reject_duplicate_names((one, two))
    assert accepted == ()
    assert set(rejected) == {one.description_url, two.description_url}
    assert all(value == ("duplicate-normalized-name",) for value in rejected.values())


def test_source_release_manifest_has_frozen_bytes_and_digest():
    root = catalogue.ROOT_URL
    page = catalogue.canonical_url(root, "/dataset.php")
    encoded, digest = catalogue.source_release_manifest((
        (page, page, b"second"),
        (root, root, b"first"),
    ))
    assert encoded == (
        b'[["https://www.timeseriesclassification.com/",'
        b'"https://www.timeseriesclassification.com/",'
        b'"a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e"],'
        b'["https://www.timeseriesclassification.com/dataset.php",'
        b'"https://www.timeseriesclassification.com/dataset.php",'
        b'"16367aacb67a4a017c8da8ab95682ccb390863780f7114dda0a0e0c55644c7c4"]]'
    )
    assert digest == hashlib.sha256(encoded).hexdigest()


def test_parser_module_has_no_network_client():
    source = Path(catalogue.__file__).read_text(encoding="utf-8")
    assert "urllib.request" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "aiohttp" not in source


def test_field_label_normalization_uses_unicode_alphanumeric_only():
    assert catalogue.normalize_field_label(" Number_of---Classes ") == "number of classes"
    assert catalogue.normalize_field_label("Caf\u00e9 / Count") == "caf\u00e9 count"


def test_anchor_text_fallback_accepts_generic_data_filenames():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL,
        "/description.php?Dataset=FixtureSeries",
    )
    body = candidate_html().replace(
        b'/data/FixtureSeries_TRAIN.tsv">TRAIN',
        b'/download/data-a.tsv">Training data',
    ).replace(
        b'/data/FixtureSeries_TEST.tsv">TEST',
        b'/download/data-b.tsv">Testing data',
    )
    result = catalogue.parse_candidate_page(url, body)
    assert result.rejections == ()
    assert result.metadata is not None
    assert result.metadata.train_url.endswith("/download/data-a.tsv")
    assert result.metadata.test_url.endswith("/download/data-b.tsv")


def test_candidate_page_failures_are_fatal_but_metadata_rejections_are_not():
    page_failure = catalogue.CandidateParseResult(
        "https://www.timeseriesclassification.com/description.php?Dataset=Broken",
        None,
        ("page:invalid UTF-8",),
    )
    with pytest.raises(RuntimeError):
        catalogue.require_no_candidate_page_failures((page_failure.description_url,), (page_failure,))
    metadata_rejection = catalogue.CandidateParseResult(
        "https://www.timeseriesclassification.com/description.php?Dataset=Missing",
        None,
        ("missing:class_count",),
    )
    catalogue.require_no_candidate_page_failures((metadata_rejection.description_url,), (metadata_rejection,))


def test_hidden_script_and_style_text_is_not_metadata():
    url=catalogue.canonical_url(catalogue.ROOT_URL, '/description.php?Dataset=FixtureSeries')
    body=candidate_html().replace(b'<html><body>', b'<html><body><script>Classes: 99</script><style>Length: 999</style>')
    result=catalogue.parse_candidate_page(url, body)
    assert result.metadata is not None
    assert result.metadata.class_count == 3
    assert result.metadata.series_length == 24


def test_candidate_result_coverage_cannot_omit_discovered_url():
    expected=(catalogue.canonical_url(catalogue.ROOT_URL, '/description.php?Dataset=Missing'),)
    with pytest.raises(RuntimeError, match='coverage mismatch'):
        catalogue.require_no_candidate_page_failures(expected, ())
