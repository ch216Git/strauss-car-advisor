import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.errors import RulesConfigError
from app.rules import DEFAULT_RULES_PATH, TOP_LEVEL, Rules, TaxBracket, load_rules

# Values copied by hand from the challenge workbook ('כללים ונתונים', 'קטלוג רכבים',
# 'חישובים'); they act as an independent oracle for the parsing, not as business logic.
WORKBOOK_CREDIT_POINT_VALUE = 242
WORKBOOK_ELECTRIC_NET_BONUS = 250
WORKBOOK_FAMILY_CARS = ("CAR-19", "CAR-20")
WORKBOOK_GROUP_B_BUDGET = 3700  # E-002: (6,000 - budget) * 50% = 1,150


def raw_rules() -> dict:
    return json.loads(DEFAULT_RULES_PATH.read_text(encoding="utf-8"))


def write_rules(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules()


# --- Loading -----------------------------------------------------------------

def test_real_rules_file_loads(rules: Rules) -> None:
    assert isinstance(rules, Rules)
    assert rules.version


def test_group_b_budget(rules: Rules) -> None:
    assert rules.budgets["ב"] == WORKBOOK_GROUP_B_BUDGET


def test_credit_point_value(rules: Rules) -> None:
    assert rules.net_method.credit_point_value == WORKBOOK_CREDIT_POINT_VALUE


def test_electric_net_bonus(rules: Rules) -> None:
    assert rules.electric.net_bonus == WORKBOOK_ELECTRIC_NET_BONUS


def test_family_track_cars(rules: Rules) -> None:
    assert rules.family_track.cars == WORKBOOK_FAMILY_CARS


def test_values_match_json(rules: Rules) -> None:
    raw = raw_rules()

    assert dict(rules.budgets) == raw["budgets"]
    assert list(rules.group_order) == raw["groupOrder"]
    assert rules.family_track.gap_split_employee == raw["familyTrack"]["gapSplitEmployee"]
    assert rules.net_method.national_insurance.ceiling == raw["netMethod"]["nationalInsurance"]["ceiling"]
    assert rules.net_method.health.high_rate == raw["netMethod"]["health"]["highRate"]


def test_brackets_keep_order_and_open_top_bracket(rules: Rules) -> None:
    raw_brackets = raw_rules()["netMethod"]["brackets"]

    assert rules.net_method.brackets == tuple(TaxBracket(limit, rate) for limit, rate in raw_brackets)
    assert rules.net_method.brackets[-1].upper_limit is None


def test_drive_field_keys_are_int_codes(rules: Rules) -> None:
    assert rules.value_of_use.drive_field == {
        int(code): name for code, name in raw_rules()["valueOfUse"]["driveField"].items()
    }


def test_loaded_once(rules: Rules) -> None:
    assert load_rules() is rules
    assert load_rules(DEFAULT_RULES_PATH) is rules


# --- Immutability ------------------------------------------------------------

def test_rules_are_immutable(rules: Rules) -> None:
    with pytest.raises(FrozenInstanceError):
        rules.version = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rules.electric.net_bonus = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        rules.budgets["א"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        rules.value_of_use.drive_field[0] = "x"  # type: ignore[index]


# --- Configuration errors ----------------------------------------------------

# Every config error carries the file, the exact key path, what was expected and what was found.

def load_error(path: Path) -> RulesConfigError:
    with pytest.raises(RulesConfigError) as info:
        load_rules(path)
    error = info.value
    assert error.file == path.resolve()
    assert str(error) == f"{path.resolve()}: {error.path}: expected {error.expected}, found {error.found}"
    return error


def broken_rules_error(tmp_path: Path, mutate) -> RulesConfigError:
    """Break one thing in a copy of the real file; rules.json itself is never changed."""
    data = raw_rules()
    mutate(data)
    return load_error(write_rules(tmp_path / "rules.json", data))


def test_message_format(tmp_path: Path) -> None:
    data = raw_rules()
    data["netMethod"]["brackets"][3][1] = 1.4
    path = write_rules(tmp_path / "rules.json", data)

    assert str(load_error(path)) == f"{path.resolve()}: netMethod.brackets[3][1]: expected rate 0..1, found 1.4"


def test_missing_file_raises(tmp_path: Path) -> None:
    error = load_error(tmp_path / "nope.json")

    assert (error.path, error.expected, error.found) == (TOP_LEVEL, "an existing rules file", "no such file")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")

    error = load_error(path)

    assert (error.path, error.expected) == (TOP_LEVEL, "valid JSON")
    assert "line 1, column 3" in error.found


def test_non_utf8_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    path.write_bytes('{"budgets": {"א": 3500}}'.encode("windows-1255"))

    error = load_error(path)

    assert (error.path, error.expected) == (TOP_LEVEL, "UTF-8 text")


def test_top_level_not_an_object_raises(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    path.write_text("[1, 2]", encoding="utf-8")

    error = load_error(path)

    assert (error.path, error.expected, error.found) == (TOP_LEVEL, "an object", "[1, 2]")


@pytest.mark.parametrize(
    "section", ["budgets", "electric", "familyTrack", "waiver", "netMethod", "valueOfUse"]
)
def test_missing_section_raises(tmp_path: Path, section: str) -> None:
    error = broken_rules_error(tmp_path, lambda d: d.pop(section))

    assert (error.path, error.expected, error.found) == (section, "a required key", "nothing (key is missing)")


def test_missing_nested_key_raises(tmp_path: Path) -> None:
    error = broken_rules_error(tmp_path, lambda d: d["netMethod"].pop("creditPointValue"))

    assert error.path == "netMethod.creditPointValue"
    assert error.found == "nothing (key is missing)"


def last_bracket_limit_as(limit):
    return lambda d: d["netMethod"]["brackets"][-1].__setitem__(0, limit)


@pytest.mark.parametrize(
    "mutate, path, expected, found",
    [
        # wrong JSON type
        (lambda d: d["budgets"].update({"א": "3500"}), "budgets.א", "an integer", '"3500"'),
        (lambda d: d["electric"].update(netBonus=True), "electric.netBonus", "an integer", "true"),
        (lambda d: d["netMethod"]["brackets"].__setitem__(2, [16150]), "netMethod.brackets[2]", "[upper limit or null, rate]", "[16150]"),
        (lambda d: d["netMethod"]["brackets"][0].__setitem__(1, "10%"), "netMethod.brackets[0][1]", "a number", '"10%"'),
        (lambda d: d["familyTrack"].update(cars="CAR-19"), "familyTrack.cars", "a list", '"CAR-19"'),
        (lambda d: d["valueOfUse"]["driveField"].update({"x": "y"}), "valueOfUse.driveField.x", "an integer code as key", '"x"'),
        # value-of-use lookup key and drive codes
        (lambda d: d["valueOfUse"].update(key=[]), "valueOfUse.key", "a non-empty list", "[]"),
        (lambda d: d["valueOfUse"]["key"].__setitem__(1, "שנה"), "valueOfUse.key[1]", "a column name known to csv_columns.COLUMN_TO_FIELD", '"שנה"'),
        (lambda d: d["valueOfUse"]["driveField"].update({"4": "בנזין"}), "valueOfUse.driveField.4", "a drive name not used by another code", '"בנזין"'),
        # every group has a budget > 0
        (lambda d: d["budgets"].pop("ג"), "budgets", 'a budget for every group in groupOrder ["א", "ב", "ג", "ד"]', 'budgets for ["א", "ב", "ד"]'),
        (lambda d: d["budgets"].update({"א": 0}), "budgets.א", "an integer > 0", "0"),
        (lambda d: d["budgets"].update({"ד": -5200}), "budgets.ד", "an integer > 0", "-5200"),
        (lambda d: d.update(groupOrder=[]), "groupOrder", "a non-empty list", "[]"),
        # positive values
        (lambda d: d["electric"].update(netBonus=0), "electric.netBonus", "an integer > 0", "0"),
        (lambda d: d["netMethod"].update(creditPointValue=-242), "netMethod.creditPointValue", "an integer > 0", "-242"),
        # non-empty family cars
        (lambda d: d["familyTrack"].update(cars=[]), "familyTrack.cars", "a non-empty list", "[]"),
        # rates between 0 and 1
        (lambda d: d["netMethod"]["brackets"][0].__setitem__(1, 10), "netMethod.brackets[0][1]", "rate 0..1", "10"),
        (lambda d: d["netMethod"]["nationalInsurance"].update(highRate=7), "netMethod.nationalInsurance.highRate", "rate 0..1", "7"),
        (lambda d: d["netMethod"]["health"].update(lowRate=-0.01), "netMethod.health.lowRate", "rate 0..1", "-0.01"),
        (lambda d: d["netMethod"]["pensionCredit"].update(rate=35), "netMethod.pensionCredit.rate", "rate 0..1", "35"),
        (lambda d: d["netMethod"]["pensionCredit"].update(maxContributionRate=1.07), "netMethod.pensionCredit.maxContributionRate", "rate 0..1", "1.07"),
        (lambda d: d["familyTrack"].update(gapSplitEmployee=50), "familyTrack.gapSplitEmployee", "rate 0..1", "50"),
        # brackets
        (lambda d: d["netMethod"].update(brackets=[]), "netMethod.brackets", "a non-empty list", "[]"),
        (lambda d: d["netMethod"]["brackets"][2].__setitem__(0, 10060), "netMethod.brackets[2][0]", "a limit above the previous one (10060)", "10060"),
        (lambda d: d["netMethod"]["brackets"][1].__setitem__(0, 5000), "netMethod.brackets[1][0]", "a limit above the previous one (7010)", "5000"),
        (lambda d: d["netMethod"]["brackets"][3].__setitem__(0, None), "netMethod.brackets[3][0]", "a limit (only the last bracket may be null)", "null"),
    ],
)
def test_invalid_value_raises(tmp_path: Path, mutate, path: str, expected: str, found: str) -> None:
    error = broken_rules_error(tmp_path, mutate)

    assert (error.path, error.expected, error.found) == (path, expected, found)


def test_long_found_value_is_shortened(tmp_path: Path) -> None:
    error = broken_rules_error(tmp_path, lambda d: d.update(electric=list(range(100))))

    assert error.path == "electric"
    assert len(error.found) == 80 and error.found.endswith("...")


@pytest.mark.parametrize("mutate", [lambda d: None, last_bracket_limit_as(80000)])
def test_valid_edge_cases_load(tmp_path: Path, mutate) -> None:
    # The unchanged file, and a last bracket with a finite limit ("may" be null, not "must").
    data = raw_rules()
    mutate(data)

    load_rules(write_rules(tmp_path / "rules.json", data))
