import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from main import ScreenContent, _itunes_upsize, _title_qualifies, fetch_poster, itunes_poster

SEARCH_URL = "https://itunes.apple.com/search?term=%s&limit=25"


def test_itunes_upsize_replaces_100x100_with_larger_box():
    url = "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/ab/cd/ef/100x100bb.jpg"
    assert _itunes_upsize(url) == (
        "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/ab/cd/ef/1200x1200bb.jpg"
    )


def test_itunes_upsize_respects_custom_box_size():
    url = "https://example.com/100x100bb.jpg"
    assert _itunes_upsize(url, box=600) == "https://example.com/600x600bb.jpg"


def test_title_qualifies_movie_needs_exact_track():
    assert _title_qualifies("feature-movie", "Interstellar", "", "Interstellar") is True
    # must NOT latch onto a same-name documentary or a sequel
    assert _title_qualifies("feature-movie", "Oppenheimer: The Real Story", "", "Oppenheimer") is False
    assert _title_qualifies("feature-movie", "Dune: Part Two", "", "Dune") is False
    assert _title_qualifies("feature-movie", "Good Boys", "", "Stranger Things") is False


def test_title_qualifies_tv_matches_show_via_collection_not_episode_title():
    # anime/TV season name lives in the collection
    assert _title_qualifies("tv-episode", "Execution", "Jujutsu Kaisen, Season 2", "Jujutsu Kaisen") is True
    # REGRESSION: an episode of an UNRELATED show titled "Stranger Things" must NOT qualify
    assert _title_qualifies("tv-episode", "Stranger Things", "Nightwatch, Season 3", "Stranger Things") is False
    assert _title_qualifies("tv-episode", "Stranger Things", "Dead by Dawn, Season 1", "Stranger Things") is False


def test_itunes_poster_returns_upsized_artwork_for_matching_movie(httpx_mock):
    httpx_mock.add_response(
        url=SEARCH_URL % "Interstellar",
        json={"results": [{
            "kind": "feature-movie", "trackName": "Interstellar",
            "releaseDate": "2014-11-05T00:00:00Z",
            "artworkUrl100": "https://example.com/is1/100x100bb.jpg",
        }]},
    )
    content = ScreenContent(content_type="movie", title="Interstellar", year=2014,
                             confidence="high", detail="")
    poster = asyncio.run(itunes_poster(content))
    assert poster == "https://example.com/is1/1200x1200bb.jpg"


def test_itunes_poster_prefers_matching_release_year(httpx_mock):
    httpx_mock.add_response(
        url=SEARCH_URL % "Dune",
        json={"results": [
            {"kind": "feature-movie", "trackName": "Dune", "releaseDate": "2021-10-22T00:00:00Z",
             "artworkUrl100": "https://example.com/2021/100x100bb.jpg"},
            {"kind": "feature-movie", "trackName": "Dune", "releaseDate": "1984-12-14T00:00:00Z",
             "artworkUrl100": "https://example.com/1984/100x100bb.jpg"},
        ]},
    )
    content = ScreenContent(content_type="movie", title="Dune", year=1984,
                             confidence="high", detail="")
    poster = asyncio.run(itunes_poster(content))
    assert poster == "https://example.com/1984/1200x1200bb.jpg"


def test_itunes_poster_picks_exact_movie_over_wrong_kind_and_superstrings(httpx_mock):
    httpx_mock.add_response(
        url=SEARCH_URL % "Oppenheimer",
        json={"results": [
            {"kind": "feature-movie", "trackName": "Oppenheimer: The Real Story",
             "releaseDate": "2023-01-01T00:00:00Z",
             "artworkUrl100": "https://example.com/doc/100x100bb.jpg"},
            {"kind": "podcast", "trackName": "Oppenheimer",
             "artworkUrl100": "https://example.com/podcast/100x100bb.jpg"},
            {"kind": "feature-movie", "trackName": "Oppenheimer",
             "releaseDate": "2023-07-21T00:00:00Z",
             "artworkUrl100": "https://example.com/right/100x100bb.jpg"},
        ]},
    )
    content = ScreenContent(content_type="movie", title="Oppenheimer", year=2023,
                             confidence="high", detail="")
    poster = asyncio.run(itunes_poster(content))
    assert poster == "https://example.com/right/1200x1200bb.jpg"


def test_itunes_poster_returns_none_rather_than_a_same_name_documentary(httpx_mock):
    # Real-world case: the actual film isn't in the catalog, only a documentary
    # sharing the name. We must return None, not the documentary's poster.
    httpx_mock.add_response(
        url=SEARCH_URL % "Oppenheimer",
        json={"results": [
            {"kind": "feature-movie", "trackName": "Oppenheimer: The Real Story",
             "releaseDate": "2023-01-01T00:00:00Z",
             "artworkUrl100": "https://example.com/doc/100x100bb.jpg"},
            {"kind": "feature-movie", "trackName": "Fat Man & Little Boy",
             "releaseDate": "1989-01-01T00:00:00Z",
             "artworkUrl100": "https://example.com/wrong/100x100bb.jpg"},
        ]},
    )
    content = ScreenContent(content_type="movie", title="Oppenheimer", year=2023,
                             confidence="high", detail="")
    assert asyncio.run(itunes_poster(content)) is None


def test_itunes_poster_matches_anime_via_collection_name(httpx_mock):
    httpx_mock.add_response(
        url=SEARCH_URL % "Jujutsu+Kaisen",
        json={"results": [
            {"kind": "tv-episode", "trackName": "Execution",
             "collectionName": "Jujutsu Kaisen, Season 2",
             "artworkUrl100": "https://example.com/jjk/100x100bb.jpg"},
        ]},
    )
    content = ScreenContent(content_type="anime", title="Jujutsu Kaisen",
                             confidence="high", detail="")
    poster = asyncio.run(itunes_poster(content))
    assert poster == "https://example.com/jjk/1200x1200bb.jpg"


def test_itunes_poster_returns_none_when_no_confident_match(httpx_mock):
    httpx_mock.add_response(
        url=SEARCH_URL % "Totally+Made+Up+Title",
        json={"results": [
            {"kind": "feature-movie", "trackName": "Something Unrelated",
             "artworkUrl100": "https://example.com/nope/100x100bb.jpg"},
        ]},
    )
    content = ScreenContent(content_type="movie", title="Totally Made Up Title",
                             confidence="high", detail="")
    assert asyncio.run(itunes_poster(content)) is None


def test_itunes_poster_returns_none_on_network_failure(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    content = ScreenContent(content_type="movie", title="Interstellar",
                             confidence="high", detail="")
    assert asyncio.run(itunes_poster(content)) is None


def test_fetch_poster_returns_itunes_result():
    content = ScreenContent(content_type="movie", title="Interstellar", year=2014,
                             confidence="high", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value="https://itunes.example/poster.jpg")):
        poster = asyncio.run(fetch_poster(content))
    assert poster == "https://itunes.example/poster.jpg"


def test_fetch_poster_returns_none_when_itunes_empty():
    content = ScreenContent(content_type="movie", title="Nonexistent",
                             confidence="low", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value=None)):
        assert asyncio.run(fetch_poster(content)) is None


def test_fetch_poster_skips_lookup_for_non_poster_content_types():
    content = ScreenContent(content_type="sports", title="Big Match",
                             confidence="medium", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value="should not be called")) as mock_itunes:
        assert asyncio.run(fetch_poster(content)) is None
    mock_itunes.assert_not_called()


def test_fetch_poster_never_raises_even_if_source_blows_up():
    content = ScreenContent(content_type="movie", title="Interstellar",
                             confidence="high", detail="")
    with patch("main.itunes_poster", AsyncMock(side_effect=RuntimeError("boom"))):
        assert asyncio.run(fetch_poster(content)) is None
