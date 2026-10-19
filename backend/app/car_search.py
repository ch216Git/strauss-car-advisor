"""Deterministic car search: what an employee types (Hebrew or English) -> catalogue car IDs.

No fuzzy matching. Input is normalized by fixed rules, mapped through the search vocabulary
(config/car_search_terms.json), and every resulting word must equal a word of the car's CSV
manufacturer, CSV commercial name or drive. Words that cannot be recognized are reported, and
then a single match is only a candidate to confirm, never an identification.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

from app.catalog import Catalog, lookup_vehicle
from app.config_file import Section, load_json_config, parse_str
from app.errors import ConfigError
from app.lookup_config import LookupConfig
from app.rules import Rules
from app.vehicle_repository import VehicleRepository

DEFAULT_TERMS_PATH = Path(__file__).parent.parent / "config" / "car_search_terms.json"

# Hebrew one-letter prefixes (ו ה ב ל מ ש כ), stripped only when the rest is a known word.
HEBREW_PREFIXES = "והבלמשכ"
_FINAL_LETTERS = str.maketrans("ךםןףץ", "כמנפצ")
_JOINERS = re.compile(r"[-'\"׳״.־]")  # hyphen, apostrophes, geresh/gershayim, dot, maqaf join word parts
_TOKEN = re.compile(r"[א-ת]+|[a-z]+|\d+")  # letters and digits split apart: rav4 -> rav 4
_YEAR = re.compile(r"(19|20)\d{2}")
_CAR_ID = re.compile(r"car\s*-?\s*(\d+)")


class MatchStatus(StrEnum):
    IDENTIFIED = "identified"  # exactly one car, every word recognized
    CANDIDATES = "candidates"  # several cars, or one car with unrecognized words: confirm with the employee
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class SearchResult:
    status: MatchStatus
    car_ids: tuple[str, ...]  # catalogue order
    unrecognized_words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SearchableCar:
    car_id: str
    registration_year: int
    words: frozenset[str]  # CSV manufacturer, CSV commercial name and drive
    name_words: frozenset[str]  # CSV commercial name only


def normalize(text: str) -> tuple[str, ...]:
    """Lower case, join hyphenated/abbreviated parts, unify Hebrew final letters, split letters from digits."""
    return tuple(_TOKEN.findall(_JOINERS.sub("", text.casefold()).translate(_FINAL_LETTERS)))


class CarSearch:
    def __init__(self, cars: Iterable[_SearchableCar], terms: Mapping[tuple[str, ...], tuple[str, ...]]) -> None:
        self._cars = tuple(cars)
        self._terms = dict(terms)
        self._vocabulary = frozenset(word for car in self._cars for word in car.words)
        self._ids_by_number = {int(car.car_id.rsplit("-", 1)[-1]): car.car_id for car in self._cars if car.car_id[-1:].isdigit()}

    def car_id(self, text: str) -> str | None:
        """A catalogue car ID written loosely ("car-10", "CAR 10", "Car10"), or None."""
        match = _CAR_ID.fullmatch(text.strip().casefold())
        return self._ids_by_number.get(int(match.group(1))) if match else None

    def search(self, query: str) -> SearchResult:
        exact_id = self.car_id(query)
        if exact_id:
            return SearchResult(MatchStatus.IDENTIFIED, (exact_id,), ())

        words, years, unrecognized = self._interpret(normalize(query))
        if not words and not years:
            return SearchResult(MatchStatus.NOT_FOUND, (), unrecognized)
        car_ids = tuple(
            car.car_id
            for car in self._cars
            if words <= car.words and (not years or car.registration_year in years)
        )
        if not car_ids:
            status = MatchStatus.NOT_FOUND
        elif len(car_ids) == 1 and not unrecognized and words:
            status = MatchStatus.IDENTIFIED
        else:
            status = MatchStatus.CANDIDATES
        return SearchResult(status, car_ids, unrecognized)

    def mentioned(self, text: str, among: Collection[str] | None = None) -> frozenset[str]:
        """Cars (from among, or the whole catalogue) that text deterministically names.

        A car is named when every word of its commercial name is in the text (after the same
        normalization and vocabulary as search) - unless another named car's name contains it,
        so "קורולה קרוס" names the COROLLA CROSS and not also the COROLLA. A car is also named
        when it is the only one (from among) that has every recognized word of the text, e.g.
        "קיה חשמלית"; a year narrows that, when it leaves any car.
        """
        cars = [car for car in self._cars if among is None or car.car_id in among]
        words, years, _ = self._interpret(normalize(text))

        named = [car for car in cars if car.name_words and car.name_words <= words]
        named = [car for car in named if not any(car.name_words < other.name_words for other in named)]
        result = {car.car_id for car in named}

        if words or among is not None:
            having_all = [car for car in cars if words <= car.words]
            in_year = [car for car in having_all if car.registration_year in years]
            only = in_year or having_all
            if len(only) == 1:
                result.add(only[0].car_id)
        return frozenset(result)

    def check_terms(self, file: Path) -> None:
        """Every vocabulary target must be words of some catalogue car."""
        for source, target in self._terms.items():
            unknown = [word for word in target if word not in self._vocabulary]
            if unknown:
                raise ConfigError(
                    f"terms.{' '.join(source)}", "words of a catalogue car's CSV names or drive", repr(unknown), file=file
                )

    def _interpret(self, tokens: tuple[str, ...]) -> tuple[frozenset[str], frozenset[int], tuple[str, ...]]:
        words: set[str] = set()
        years: set[int] = set()
        unrecognized: list[str] = []
        i = 0
        while i < len(tokens):
            term_length, target = self._term_at(tokens, i)
            if term_length:
                words.update(target)
                i += term_length
                continue
            token = tokens[i]
            i += 1
            if _YEAR.fullmatch(token):
                years.add(int(token))
            elif token in self._vocabulary:
                words.add(token)
            elif (known := self._without_prefix(token)) is not None:
                words.update(known)
            elif len(token) == 1 and token in HEBREW_PREFIXES:
                continue  # a lone preposition, e.g. "ל" of "ל-BYD"
            else:
                unrecognized.append(token)
        return frozenset(words), frozenset(years), tuple(unrecognized)

    def _term_at(self, tokens: tuple[str, ...], i: int) -> tuple[int, tuple[str, ...]]:
        """Longest vocabulary term starting at tokens[i], also with a Hebrew prefix on its first word."""
        for length in range(len(tokens) - i, 0, -1):
            phrase = tokens[i : i + length]
            for first in (phrase[0], *prefix_stripped(phrase[0])):
                target = self._terms.get((first, *phrase[1:]))
                if target:
                    return length, target
        return 0, ()

    def _without_prefix(self, token: str) -> tuple[str, ...] | None:
        for stripped in prefix_stripped(token):
            if stripped in self._vocabulary:
                return (stripped,)
        return None


def prefix_stripped(token: str) -> tuple[str, ...]:
    """token without 1 or 2 leading Hebrew prefix letters, keeping at least 2 letters."""
    return tuple(
        token[k:] for k in (1, 2) if len(token) - k >= 2 and all(letter in HEBREW_PREFIXES for letter in token[:k])
    )


def build_car_search(
    catalog: Catalog,
    repository: VehicleRepository,
    lookup: LookupConfig,
    rules: Rules,
    terms_path: Path = DEFAULT_TERMS_PATH,
) -> CarSearch:
    cars = []
    for car in catalog.cars.values():
        record = lookup_vehicle(car, repository, lookup)
        drive = rules.value_of_use.drive_field.get(record.propulsion_code, "")
        words = frozenset(normalize(f"{record.manufacturer} {record.commercial_name} {drive}"))
        name_words = frozenset(normalize(record.commercial_name))
        cars.append(_SearchableCar(car.car_id, car.registration_year, words, name_words))
    search = CarSearch(cars, load_search_terms(terms_path))
    search.check_terms(terms_path.resolve())
    return search


def load_search_terms(path: Path = DEFAULT_TERMS_PATH) -> Mapping[tuple[str, ...], tuple[str, ...]]:
    return _load_terms(path.resolve())


@cache
def _load_terms(path: Path) -> Mapping[tuple[str, ...], tuple[str, ...]]:
    return load_json_config(path, _parse_terms)


def _parse_terms(root: Section) -> Mapping[tuple[str, ...], tuple[str, ...]]:
    terms: dict[tuple[str, ...], tuple[str, ...]] = {}
    for source, target in root.mapping_of("terms", parse_str).items():
        key, value = normalize(source), normalize(target)
        if not key or not value:
            raise ConfigError(f"terms.{source}", "a key and target with letters or digits", repr(target))
        if key in terms:
            raise ConfigError(f"terms.{source}", "a key that differs from the others after normalization", repr(source))
        terms[key] = value
    return terms
