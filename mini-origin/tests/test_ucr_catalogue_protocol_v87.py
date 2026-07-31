from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mini_origin import ucr_catalogue_protocol_v87 as catalogue


def page_attempt(
    index: int,
    url: str,
    body: bytes | None,
    *,
    failure: str | None = None,
    final_url: str | None = None,
    status_code: int | None = None,
    tls_validated: bool | None = None,
    redirects: tuple[str, ...] = (),
    connect_times: tuple[float, ...] | None = None,
    read_time: float | None = None,
    total_time: float | None = None,
) -> catalogue.HTMLPageAttempt:
    success = failure is None
    if connect_times is None:
        connect_times = (0.1,) * (len(redirects) + 1) if success else ()
    return catalogue.HTMLPageAttempt(
        attempt_index=index,
        scheduled_delay_seconds=catalogue.HTML_PAGE_RETRY_DELAYS_SECONDS[index - 1],
        request_method=catalogue.HTML_PAGE_REQUEST_METHOD,
        request_headers=catalogue.HTML_PAGE_REQUEST_HEADERS,
        tls_certificate_validated=(success if tls_validated is None else tls_validated),
        redirect_chain=redirects,
        connect_elapsed_seconds=connect_times,
        read_elapsed_seconds=(0.2 if success and read_time is None else read_time),
        total_elapsed_seconds=(0.4 if success and total_time is None else total_time),
        status_code=(200 if success and status_code is None else status_code),
        final_url=(url if success and final_url is None else final_url),
        body=body,
        failure=failure,
    )


def authoritative_page(url: str, body: bytes) -> catalogue.AuthoritativeHTMLPage:
    return catalogue.authoritative_html_page(url, (
        page_attempt(1, url, body),
        page_attempt(2, url, None, failure="timeout"),
        page_attempt(3, url, body),
    ))


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


def test_authoritative_page_rejects_redirect_chain_final_url_mismatch():
    requested = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html?view=all")
    redirect = catalogue.canonical_url(catalogue.ROOT_URL, "/archive-v2.html?view=all")
    wrong_final = catalogue.canonical_url(catalogue.ROOT_URL, "/archive-v3.html?view=all")
    attempts = tuple(
        page_attempt(
            index,
            requested,
            b"ok",
            redirects=(redirect,),
            final_url=wrong_final,
        )
        for index in catalogue.HTML_PAGE_ATTEMPT_INDICES
    )
    with pytest.raises(ValueError, match="redirect-chain endpoint"):
        catalogue.authoritative_html_page(requested, attempts)


def test_authoritative_page_rejects_unrecorded_final_url_without_redirect():
    requested = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html?view=all")
    wrong_final = catalogue.canonical_url(catalogue.ROOT_URL, "/archive-v2.html?view=all")
    attempts = tuple(
        page_attempt(index, requested, b"ok", final_url=wrong_final)
        for index in catalogue.HTML_PAGE_ATTEMPT_INDICES
    )
    with pytest.raises(ValueError, match="redirect-chain endpoint"):
        catalogue.authoritative_html_page(requested, attempts)


def test_authoritative_page_rejects_failed_attempt_with_success_state_fields():
    requested = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html?view=all")
    attempts = (
        page_attempt(
            1,
            requested,
            None,
            failure="timeout",
            final_url=requested,
            status_code=200,
        ),
        page_attempt(2, requested, None, failure="timeout"),
        page_attempt(3, requested, None, failure="timeout"),
    )
    with pytest.raises(ValueError, match="success-state fields"):
        catalogue.authoritative_html_page(requested, attempts)


def test_authoritative_page_rejects_failed_attempt_with_transport_success_evidence():
    requested = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html?view=all")
    contradictory_attempts = (
        page_attempt(1, requested, None, failure="timeout", tls_validated=True),
        page_attempt(1, requested, None, failure="timeout", redirects=(requested,)),
        page_attempt(1, requested, None, failure="timeout", connect_times=(0.1,)),
        page_attempt(1, requested, None, failure="timeout", read_time=0.2),
        page_attempt(1, requested, None, failure="timeout", total_time=0.4),
    )
    for contradictory in contradictory_attempts:
        attempts = (
            contradictory,
            page_attempt(2, requested, None, failure="timeout"),
            page_attempt(3, requested, None, failure="timeout"),
        )
        with pytest.raises(ValueError, match="transport-success evidence"):
            catalogue.authoritative_html_page(requested, attempts)


@pytest.mark.parametrize("failure", ("", "   ", " timeout", "timeout\n"))
def test_authoritative_page_rejects_noncanonical_failure_text(failure: str):
    requested = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html?view=all")
    attempts = tuple(
        page_attempt(index, requested, None, failure=failure)
        for index in catalogue.HTML_PAGE_ATTEMPT_INDICES
    )
    with pytest.raises(ValueError, match="normalized non-empty failure text"):
        catalogue.authoritative_html_page(requested, attempts)


def test_authoritative_page_rejects_impossible_total_timing():
    requested = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html?view=all")
    attempts = tuple(
        page_attempt(
            index,
            requested,
            b"ok",
            connect_times=(0.2,),
            read_time=0.3,
            total_time=0.49,
        )
        for index in catalogue.HTML_PAGE_ATTEMPT_INDICES
    )
    with pytest.raises(ValueError, match="shorter than recorded transport phases"):
        catalogue.authoritative_html_page(requested, attempts)


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
        root: authoritative_page(root, (
            '<a href="/dataset.php">datasets</a>'
            '<a href="/archive.html">archive</a>'
        ).encode()),
        listing_a: authoritative_page(
            listing_a, f'<a href="{candidate_a}">Beta</a>'.encode()
        ),
        listing_b: authoritative_page(
            listing_b, f'<a href="{candidate_b}">Alpha</a>'.encode()
        ),
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
    first = authoritative_page(root, b"first")
    second = catalogue.authoritative_html_page(page, (
        page_attempt(1, page, b"second"),
        page_attempt(2, page, b"second"),
        page_attempt(3, page, None, failure="HTTP 503"),
    ))
    encoded, digest = catalogue.source_release_manifest(
        (root, page), (), {root: first, page: second}
    )
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
        catalogue.require_no_candidate_page_failures((page_failure,))
    metadata_rejection = catalogue.CandidateParseResult(
        "https://www.timeseriesclassification.com/description.php?Dataset=Missing",
        None,
        ("missing:class_count",),
    )
    catalogue.require_no_candidate_page_failures((metadata_rejection,))


def test_every_html_page_uses_exact_three_attempt_authority_and_agreement():
    url = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html")
    page = catalogue.authoritative_html_page(url, (
        page_attempt(3, url, b"same"),
        page_attempt(1, url, b"same"),
        page_attempt(2, url, None, failure="timeout"),
    ))
    assert page.authoritative_attempt_index == 1
    assert page.body == b"same"
    assert page.body_sha256 == hashlib.sha256(b"same").hexdigest()
    with pytest.raises(ValueError):
        catalogue.authoritative_html_page(url, (
            page_attempt(1, url, b"same"),
            page_attempt(2, url, b"same"),
        ))
    with pytest.raises(RuntimeError):
        catalogue.authoritative_html_page(url, (
            page_attempt(1, url, b"one"),
            page_attempt(2, url, b"two"),
            page_attempt(3, url, None, failure="timeout"),
        ))
    with pytest.raises(ValueError):
        catalogue.authoritative_html_page(url, (
            page_attempt(
                1, url, b"x" * (catalogue.HTML_PAGE_MAX_RESPONSE_BYTES + 1)
            ),
            page_attempt(2, url, None, failure="timeout"),
            page_attempt(3, url, None, failure="timeout"),
        ))
    with pytest.raises(ValueError):
        catalogue.authoritative_html_page(url, (
            page_attempt(1, url, b"same", read_time=61.0),
            page_attempt(2, url, None, failure="timeout"),
            page_attempt(3, url, None, failure="timeout"),
        ))


def test_candidate_results_must_exactly_cover_discovered_urls():
    one_url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=One"
    )
    two_url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=Two"
    )
    one = catalogue.CandidateParseResult(one_url, None, ("missing:class_count",))
    two = catalogue.CandidateParseResult(two_url, None, ("missing:class_count",))
    catalogue.validate_candidate_result_coverage((one_url, two_url), (one, two))
    with pytest.raises(RuntimeError):
        catalogue.validate_candidate_result_coverage((one_url, two_url), (one,))
    with pytest.raises(RuntimeError):
        catalogue.validate_candidate_result_coverage((one_url,), (one, two))
    with pytest.raises(ValueError):
        catalogue.validate_candidate_result_coverage((one_url,), (one, one))
    failed = catalogue.CandidateParseResult(two_url, None, ("page:invalid UTF-8",))
    with pytest.raises(RuntimeError):
        catalogue.validate_candidate_result_coverage((one_url, two_url), (one, failed))


def test_numeric_series_eligibility_is_metadata_only_and_exact():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=FixtureSeries"
    )
    result = catalogue.parse_candidate_page(url, candidate_html())
    assert result.metadata is not None
    assert catalogue.metadata_numeric_series_eligible(result.metadata) is True
    assert catalogue.metadata_eligibility_rejections(result.metadata) == ()
    bad = catalogue.CandidateMetadata(
        normalized_dataset_name=result.metadata.normalized_dataset_name,
        total_instances=299,
        series_length=result.metadata.series_length,
        class_count=result.metadata.class_count,
        classification=True,
        univariate=True,
        train_url="https://www.timeseriesclassification.com/data/train.bin",
        test_url=result.metadata.test_url,
        description_url=result.metadata.description_url,
        metadata_page_sha256=result.metadata.metadata_page_sha256,
    )
    assert catalogue.metadata_eligibility_rejections(bad) == (
        "numeric-series-metadata-predicate-failed",
        "total-instances-outside-range",
    )


def test_script_style_and_other_non_visible_text_do_not_create_metadata_conflicts():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=FixtureSeries"
    )
    body = candidate_html().replace(
        b"<html><body>",
        b"<html><head><style>Classes: 98</style><script>Classes: 99</script></head><body>"
        b"<noscript>Train size: 777</noscript><template>Test size: 888</template>",
    )
    result = catalogue.parse_candidate_page(url, body)
    assert result.rejections == ()
    assert result.metadata is not None
    assert result.metadata.class_count == 3
    assert result.metadata.total_instances == 321


def test_source_manifest_is_the_only_frozen_source_release_digest():
    root = catalogue.ROOT_URL
    page = catalogue.authoritative_html_page(root, (
        page_attempt(1, root, b"root"),
        page_attempt(2, root, b"root"),
        page_attempt(3, root, b"root"),
    ))
    encoded, frozen_source_release = catalogue.source_release_manifest(
        (root,), (), {root: page}
    )
    assert frozen_source_release == hashlib.sha256(encoded).hexdigest()
    assert frozen_source_release != page.body_sha256


def test_manifest_requires_exact_root_traversal_and_candidate_page_coverage():
    root = catalogue.ROOT_URL
    candidate = catalogue.canonical_url(
        root, "/description.php?Dataset=FixtureSeries"
    )
    root_page = authoritative_page(root, b"root")
    candidate_page = authoritative_page(candidate, candidate_html())
    catalogue.source_release_manifest(
        (root,), (candidate,), {root: root_page, candidate: candidate_page}
    )
    with pytest.raises(RuntimeError):
        catalogue.source_release_manifest(
            (root,), (candidate,), {root: root_page}
        )
    with pytest.raises(RuntimeError):
        catalogue.source_release_manifest(
            (root,), (), {root: root_page, candidate: candidate_page}
        )


def test_frozen_candidate_workflow_consumes_authoritative_page_objects():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=FixtureSeries"
    )
    page = authoritative_page(url, candidate_html())
    result = catalogue.parse_authoritative_candidate_page(page)
    assert result.metadata is not None
    assert result.rejections == ()


def test_authoritative_page_is_revalidated_from_retained_attempt_evidence():
    url = catalogue.canonical_url(catalogue.ROOT_URL, "/archive.html")
    page = authoritative_page(url, b"real")
    forged = catalogue.AuthoritativeHTMLPage(
        requested_url=page.requested_url,
        final_url=page.final_url,
        body=b"forged",
        body_sha256=hashlib.sha256(b"forged").hexdigest(),
        authoritative_attempt_index=page.authoritative_attempt_index,
        attempts=page.attempts,
    )
    with pytest.raises(ValueError):
        catalogue.revalidate_authoritative_html_page(forged)


def test_finalize_metadata_snapshot_derives_complete_page_set_and_eligibility():
    root = catalogue.ROOT_URL
    listing = catalogue.canonical_url(root, "/archive.html")
    candidate = catalogue.canonical_url(
        root, "/description.php?Dataset=FixtureSeries"
    )
    pages = {
        root: authoritative_page(
            root, b'<a href="/archive.html">archive</a>'
        ),
        listing: authoritative_page(
            listing,
            f'<a href="{candidate}">candidate</a>'.encode(),
        ),
        candidate: authoritative_page(candidate, candidate_html()),
    }
    snapshot = catalogue.finalize_metadata_snapshot(
        root, pages, frozenset()
    )
    assert snapshot.crawl_page_urls == (root, listing)
    assert snapshot.candidate_urls == (candidate,)
    assert len(snapshot.eligible_metadata) == 1
    assert snapshot.rejections == ()
    assert snapshot.frozen_source_release == hashlib.sha256(
        snapshot.source_manifest_bytes
    ).hexdigest()
    excluded = catalogue.finalize_metadata_snapshot(
        root, pages, frozenset({"fixtureseries"})
    )
    assert excluded.eligible_metadata == ()
    assert excluded.rejections == ((
        candidate, ("name-in-v86-contamination-registry",)
    ),)
    with pytest.raises(KeyError):
        catalogue.finalize_metadata_snapshot(
            root, {root: pages[root], listing: pages[listing]}, frozenset()
        )


def test_duplicate_name_rejection_includes_rows_with_failed_metadata():
    one_url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=FixtureSeries"
    )
    two_url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/other/description.php?Dataset=FixtureSeries"
    )
    valid = catalogue.parse_candidate_page(one_url, candidate_html())
    invalid = catalogue.parse_candidate_page(
        two_url,
        candidate_html().replace(
            b"<tr><th>Test size</th><td>121</td></tr>", b""
        ),
    )
    accepted, rejected = catalogue.reject_duplicate_names((valid, invalid))
    assert accepted == ()
    assert rejected[one_url] == ("duplicate-normalized-name",)
    assert "duplicate-normalized-name" in rejected[two_url]
    assert "'missing:test_size'" in rejected[two_url]


def test_complete_page_cap_counts_candidate_pages(monkeypatch):
    root = catalogue.ROOT_URL
    one = catalogue.canonical_url(root, "/description.php?Dataset=One")
    two = catalogue.canonical_url(root, "/description.php?Dataset=Two")
    root_body = (
        f'<a href="{one}">one</a><a href="{two}">two</a>'
    ).encode()
    pages = {
        root: authoritative_page(root, root_body),
        one: authoritative_page(
            one, candidate_html().replace(b"FixtureSeries", b"One")
        ),
        two: authoritative_page(
            two, candidate_html().replace(b"FixtureSeries", b"Two")
        ),
    }
    monkeypatch.setattr(catalogue, "MAX_HTML_PAGES", 2)
    with pytest.raises(OverflowError):
        catalogue.finalize_metadata_snapshot(root, pages, frozenset())


def test_exclusion_registry_names_must_already_be_normalized():
    root = catalogue.ROOT_URL
    with pytest.raises(ValueError):
        catalogue.finalize_metadata_snapshot(
            root,
            {root: authoritative_page(root, b"<html></html>")},
            frozenset({"Not Normalized"}),
        )


def test_path_percent_encoding_is_decoded_normalized_and_reencoded_once():
    assert catalogue.canonical_url(
        catalogue.ROOT_URL, "/caf%C3%A9/data"
    ) == "https://www.timeseriesclassification.com/caf%C3%A9/data"
    assert catalogue.canonical_url(
        catalogue.ROOT_URL, "/caf\u0065\u0301/data"
    ) == "https://www.timeseriesclassification.com/caf%C3%A9/data"
    with pytest.raises(ValueError):
        catalogue.canonical_url(catalogue.ROOT_URL, "/bad%2/path")
    with pytest.raises(UnicodeDecodeError):
        catalogue.canonical_url(catalogue.ROOT_URL, "/bad%FF/path")


def test_crawl_resolves_relative_links_against_authoritative_final_url():
    root = catalogue.ROOT_URL
    final_root = catalogue.canonical_url(root, "/base/index.html")
    child = catalogue.canonical_url(final_root, "child.html")
    root_page = catalogue.authoritative_html_page(root, (
        page_attempt(1, root, b'<a href="child.html">child</a>', redirects=(final_root,), final_url=final_root),
        page_attempt(2, root, None, failure="timeout"),
        page_attempt(3, root, b'<a href="child.html">child</a>', redirects=(final_root,), final_url=final_root),
    ))
    pages = {
        root: root_page,
        child: authoritative_page(child, b"<html></html>"),
    }
    order, candidates = catalogue.catalogue_plan(root, pages)
    assert order == (root, child)
    assert candidates == ()


def test_candidate_relative_files_use_final_url_but_identity_cannot_change():
    requested = catalogue.canonical_url(
        catalogue.ROOT_URL, "/catalog/description.php?Dataset=FixtureSeries"
    )
    final = requested.replace("www.timeseriesclassification.com", "timeseriesclassification.com")
    body = candidate_html().replace(
        b'/data/FixtureSeries_TRAIN.tsv', b'../data/FixtureSeries_TRAIN.tsv'
    ).replace(
        b'/data/FixtureSeries_TEST.tsv', b'../data/FixtureSeries_TEST.tsv'
    )
    page = catalogue.authoritative_html_page(requested, (
        page_attempt(1, requested, body, redirects=(final,), final_url=final),
        page_attempt(2, requested, None, failure="timeout"),
        page_attempt(3, requested, body, redirects=(final,), final_url=final),
    ))
    result = catalogue.parse_authoritative_candidate_page(page)
    assert result.metadata is not None
    assert result.metadata.train_url == (
        "https://timeseriesclassification.com/data/FixtureSeries_TRAIN.tsv"
    )
    changed = final.replace("FixtureSeries", "Other")
    bad = catalogue.authoritative_html_page(requested, (
        page_attempt(1, requested, body, redirects=(changed,), final_url=changed),
        page_attempt(2, requested, None, failure="timeout"),
        page_attempt(3, requested, body, redirects=(changed,), final_url=changed),
    ))
    with pytest.raises(RuntimeError):
        catalogue.parse_authoritative_candidate_page(bad)


def test_train_and_test_urls_must_be_distinct():
    url = catalogue.canonical_url(
        catalogue.ROOT_URL, "/description.php?Dataset=FixtureSeries"
    )
    body = candidate_html().replace(
        b'/data/FixtureSeries_TRAIN.tsv">TRAIN',
        b'/data/shared.tsv">Training data',
    ).replace(
        b'/data/FixtureSeries_TEST.tsv">TEST',
        b'/data/shared.tsv">Testing data',
    )
    result = catalogue.parse_candidate_page(url, body)
    assert result.metadata is None
    assert "train-test-url-identical" in result.rejections
