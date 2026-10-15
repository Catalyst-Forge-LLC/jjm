"""Banking77 as a routing task: which of eight teams should take a bank customer's message.

Banking77 (PolyAI, CC-BY-4.0): 13,083 real customer messages, each labelled by people with one
of 77 intents. The eight teams below group those intents. The grouping is ours and fixed here;
the label on every message is still the human intent label, mapped through this table.

Download (pinned):
    https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/<COMMIT>/banking_data/{train,test}.csv
into data/banking77/, which git ignores.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from jjm.compile import compile_criterion, render_state

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "banking77"
COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SHA256 = {"train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
          "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"}

TEAMS: dict[str, tuple[str, list[str]]] = {
    "cards": ("Getting, activating, linking, or replacing a card, card delivery, PINs, virtual and disposable cards, "
              "Apple Pay or Google Pay, or a card that does not work",
              ["activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire", "card_acceptance", "card_arrival",
               "card_delivery_estimate", "card_linking", "card_not_working", "card_swallowed", "change_pin",
               "contactless_not_working", "disposable_card_limits", "get_disposable_virtual_card", "get_physical_card",
               "getting_spare_card", "getting_virtual_card", "order_physical_card", "pin_blocked",
               "virtual_card_not_working", "visa_or_mastercard"]),
    "payments": ("The status of a card payment: declined, pending, reverted, or charged twice, or a refund request "
                 "or a refund that has not arrived",
                 ["declined_card_payment", "pending_card_payment", "reverted_card_payment?", "Refund_not_showing_up",
                  "request_refund", "transaction_charged_twice"]),
    "transfers": ("Bank transfers into or out of the account: sending, receiving, cancelling, timing, failed or "
                  "declined transfers, or beneficiaries",
                  ["balance_not_updated_after_bank_transfer", "beneficiary_not_allowed", "cancel_transfer",
                   "declined_transfer", "failed_transfer", "pending_transfer", "receiving_money",
                   "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing"]),
    "top_up": ("Topping up the account by card, bank transfer, cash, or cheque, automatic top-ups, top-up limits, "
               "and failed, pending, or reverted top-ups",
               ["automatic_top_up", "balance_not_updated_after_cheque_or_cash_deposit", "pending_top_up",
                "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
                "verify_top_up"]),
    "cash": ("ATM cash withdrawals: finding or using an ATM, declined or pending withdrawals, or the wrong amount "
             "of cash received",
             ["atm_support", "declined_cash_withdrawal", "pending_cash_withdrawal", "wrong_amount_of_cash_received"]),
    "fees_fx": ("Fees and charges on payments, withdrawals, transfers, or top-ups, exchange rates and currency "
                "exchange, and supported currencies or card types",
                ["card_payment_fee_charged", "cash_withdrawal_charge", "exchange_charge", "exchange_rate",
                 "exchange_via_app", "extra_charge_on_statement", "top_up_by_bank_transfer_charge",
                 "top_up_by_card_charge", "transfer_fee_charged", "card_payment_wrong_exchange_rate",
                 "wrong_exchange_rate_for_cash_withdrawal", "fiat_currency_support",
                 "supported_cards_and_currencies"]),
    "security": ("A payment, withdrawal, or direct debit the customer does not recognise, a compromised card, or a "
                 "lost or stolen card or phone",
                 ["card_payment_not_recognised", "cash_withdrawal_not_recognised",
                  "direct_debit_payment_not_recognised", "compromised_card", "lost_or_stolen_card",
                  "lost_or_stolen_phone"]),
    "account": ("Identity verification, source of funds, personal details, passcodes, closing the account, and age "
                "or country eligibility",
                ["age_limit", "country_support", "edit_personal_details", "passcode_forgotten", "terminate_account",
                 "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "why_verify_identity"]),
}
TEAM_IDS = list(TEAMS)
INTENT_TEAM = {intent: team for team, (_, intents) in TEAMS.items() for intent in intents}
QUESTION = "Which team should handle this bank customer's message?"


def criterion() -> dict[str, Any]:
    return {"id": "route-banking", "version": "1", "kind": "choice", "question": QUESTION,
            "required_evidence_ids": [], "accept_outcomes": [],
            "options": [{"id": t, "description": TEAMS[t][0]} for t in TEAM_IDS]}


def compiled():
    return compile_criterion(criterion())


def state(message: str) -> dict[str, Any]:
    return {"content": {"message": message}, "evidence": []}


def load(split: str) -> list[dict[str, str]]:
    path = DATA / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing. See trials/routing/data.py for the pinned download.")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256[path.name]:
        raise SystemExit(f"{path} has sha256 {got}, expected {SHA256[path.name]}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [{"text": r["text"], "intent": r["category"], "team": INTENT_TEAM[r["category"]]}
                for r in csv.DictReader(fh)]
    return rows


def texts(rows: list[dict[str, str]]) -> list[str]:
    return [render_state(state(r["text"])) for r in rows]


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([TEAM_IDS.index(r["team"]) for r in rows], dtype=int)


def stratified(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        out.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    return np.array(sorted(out), dtype=int)


assert len(INTENT_TEAM) == 77 and sum(len(v[1]) for v in TEAMS.values()) == 77
