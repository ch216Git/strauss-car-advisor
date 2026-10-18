import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.errors import RulesConfigError
from app.rules import DEFAULT_RULES_PATH, Rules, TaxBracket, load_rules

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

def test_rules_file_loads(rules: Rules) -> None:
    assert rules.version


def test_values_match_workbook(rules: Rules) -> None:
    assert rules.budgets["ב"] == WORKBOOK_GROUP_B_BUDGET
    assert rules.net_method.credit_point_value == WORKBOOK_CREDIT_POINT_VALUE
    assert rules.electric.net_bonus == WORKBOOK_ELECTRIC_NET_BONUS
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

def test_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"

    with pytest.raises(RulesConfigError, match="not found"):
        load_rules(path)


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(RulesConfigError, match="not valid JSON"):
        load_rules(path)


def test_missing_nested_key_names_its_path(tmp_path: Path) -> None:
    data = raw_rules()
    del data["netMethod"]["creditPointValue"]

    with pytest.raises(RulesConfigError, match=r"missing required key 'netMethod\.creditPointValue'"):
        load_rules(write_rules(tmp_path / "rules.json", data))


@pytest.mark.parametrize(
    "mutate, where",
    [
        (lambda d: d["budgets"].update({"א": "3500"}), "budgets.א"),
        (lambda d: d["electric"].update(netBonus=True), "electric.netBonus"),
        (lambda d: d["netMethod"]["brackets"].__setitem__(2, [16150]), r"netMethod\.brackets\[2\]"),
        (lambda d: d["netMethod"]["brackets"][0].__setitem__(1, "10%"), r"netMethod\.brackets\[0\]\[1\]"),
        (lambda d: d["familyTrack"].update(cars="CAR-19"), "familyTrack.cars"),
        (lambda d: d["valueOfUse"]["driveField"].update({"x": "y"}), "valueOfUse.driveField"),
    ],
)
def test_wrong_type_names_its_path(tmp_path: Path, mutate, where: str) -> None:
    data = raw_rules()
    mutate(data)

    with pytest.raises(RulesConfigError, match=where):
        load_rules(write_rules(tmp_path / "rules.json", data))
