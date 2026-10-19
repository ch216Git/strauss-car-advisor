import json
from pathlib import Path

import pytest

from app.car_search import DEFAULT_TERMS_PATH, CarSearch, MatchStatus, build_car_search, normalize
from app.catalog import Catalog, load_catalog
from app.errors import ConfigError
from app.lookup_config import load_lookup_config
from app.rules import load_rules
from app.vehicle_repository import VehicleRepository


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="module")
def repository() -> VehicleRepository:
    return VehicleRepository()


@pytest.fixture(scope="module")
def search(catalog: Catalog, repository: VehicleRepository) -> CarSearch:
    return build_car_search(catalog, repository, load_lookup_config(), load_rules())


def ids(catalog: Catalog, *commercial_names: str) -> tuple[str, ...]:
    """Catalogue IDs by commercial name, in catalogue order (tests name cars, not IDs)."""
    return tuple(car.car_id for car in catalog.cars.values() if car.commercial_name in commercial_names)


# --- Normalization -----------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Toyota RAV-4", ("toyota", "rav", "4")),
        ("RAV4", ("rav", "4")),
        ("ספורטאז'", ("ספורטאז",)),
        ("ספורטאז׳", ("ספורטאז",)),  # Hebrew geresh
        ("פלאג-אין", ("פלאגאינ",)),  # hyphen joined, final nun unified
        ("ל-BYD", ("ל", "byd")),  # Hebrew and Latin split apart
        ("מ.ג", ("מג",)),
        ("  ", ()),
    ],
)
def test_normalize(text: str, expected: tuple[str, ...]) -> None:
    assert normalize(text) == expected


# --- Identified: Hebrew, English, mixed ----------------------------------------

@pytest.mark.parametrize(
    "query, commercial_name",
    [
        # Hebrew
        ("קורולה קרוס", "COROLLA CROSS"),
        ("טוסון", "TUCSON"),
        ("קרניבל", "CARNIVAL"),
        ("סנטה פה", "SANTA FE"),
        ("טסלה מודל 3", "MODEL 3"),
        ("אוקטביה", "OCTAVIA"),
        ("אוקטאביה", "OCTAVIA"),
        ("ספורטאז'", "SPORTAGE HEV"),
        ("מאזדה 3", "MAZDA 3"),
        ("ראב 4", "RAV4"),
        # English
        ("corolla cross", "COROLLA CROSS"),
        ("Toyota Corolla Cross 2025", "COROLLA CROSS"),
        ("BYD Atto 3", "ATTO 3"),
        ("RAV4", "RAV4"),
        ("rav 4", "RAV4"),
        ("MG4", "MG4"),
        ("ev3", "EV3"),
        ("Kona EV", "KONA EV"),
        # Mixed
        ("BYD אטו 3", "ATTO 3"),
        ("טויוטה RAV4", "RAV4"),
        # Hebrew prefixes, as in "מהקורולה קרוס ל-BYD Atto 3"
        ("האטו 3", "ATTO 3"),
        ("מהקורולה קרוס", "COROLLA CROSS"),
        ("ל-BYD Atto 3", "ATTO 3"),
        # Narrowed by year or drive
        ("קורולה 2026", "COROLLA"),
        ("קיה חשמלית", "EV3"),
        ("kia electric", "EV3"),
        ("פלאג-אין", "SPORTAGE HEV"),
    ],
)
def test_identifies_one_car(search: CarSearch, catalog: Catalog, query: str, commercial_name: str) -> None:
    result = search.search(query)

    assert result.status == MatchStatus.IDENTIFIED
    assert result.car_ids == ids(catalog, commercial_name)
    assert result.unrecognized_words == ()


def test_every_catalogue_car_is_identified_by_its_name_and_year(search: CarSearch, catalog: Catalog) -> None:
    for car in catalog.cars.values():
        result = search.search(f"{car.commercial_name} {car.registration_year}")
        assert (result.status, result.car_ids) == (MatchStatus.IDENTIFIED, (car.car_id,)), car


@pytest.mark.parametrize("text", ["car-10", "CAR 10", "Car10", "CAR-10"])
def test_loosely_written_car_id(search: CarSearch, text: str) -> None:
    assert search.car_id(text) == "CAR-10"
    assert search.search(text).car_ids == ("CAR-10",)


# --- Ambiguous: candidates, never a guess ---------------------------------------

@pytest.mark.parametrize(
    "query, commercial_names",
    [
        ("קורולה", ("COROLLA", "COROLLA CROSS")),
        ("corolla", ("COROLLA", "COROLLA CROSS")),
        ("קרוס", ("YARIS CROSS HSD", "COROLLA CROSS")),
        ("טויוטה היברידית", ("YARIS CROSS HSD", "COROLLA CROSS", "RAV4")),
    ],
)
def test_ambiguous_query_returns_candidates(search: CarSearch, catalog: Catalog, query: str, commercial_names) -> None:
    result = search.search(query)

    assert result.status == MatchStatus.CANDIDATES
    assert result.car_ids == ids(catalog, *commercial_names)


def test_unrecognized_words_turn_a_single_match_into_a_candidate(search: CarSearch, catalog: Catalog) -> None:
    # The employee drives a Mazda CX-5; only "Mazda" is known, so the Mazda 3 must be confirmed, not chosen.
    result = search.search("מאזדה CX-5")

    assert result.status == MatchStatus.CANDIDATES
    assert result.car_ids == ids(catalog, "MAZDA 3")
    assert result.unrecognized_words == ("cx", "5")


# --- Not found: no fuzzy matching ----------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "פרארי",
        "ferrari",
        "קורולא",  # misspelled: not silently read as Corolla
        "קורולה קרוס 2019",  # the right model in a year the catalogue does not have
        "car-99",
    ],
)
def test_not_found(search: CarSearch, query: str) -> None:
    result = search.search(query)

    assert (result.status, result.car_ids) == (MatchStatus.NOT_FOUND, ())


# --- Search vocabulary file ------------------------------------------------------

def write_terms(tmp_path: Path, terms: dict) -> Path:
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"terms": terms}, ensure_ascii=False), encoding="utf-8")
    return path


def test_term_target_must_exist_in_catalogue(tmp_path: Path, catalog, repository) -> None:
    path = write_terms(tmp_path, {"פרארי": "ferrari"})

    with pytest.raises(ConfigError) as info:
        build_car_search(catalog, repository, load_lookup_config(), load_rules(), path)

    assert (info.value.file, info.value.path) == (path.resolve(), "terms.פרארי")


def test_terms_must_differ_after_normalization(tmp_path: Path, catalog, repository) -> None:
    path = write_terms(tmp_path, {"Toyota": "טויוטה", "toyota": "טויוטה"})

    with pytest.raises(ConfigError, match="differs from the others"):
        build_car_search(catalog, repository, load_lookup_config(), load_rules(), path)


def test_real_terms_file_is_valid(catalog, repository) -> None:
    build_car_search(catalog, repository, load_lookup_config(), load_rules(), DEFAULT_TERMS_PATH)
