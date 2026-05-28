"""Autonomous bidding agent for the used-car auction simulation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import pandas as pd


class BiddingAgent:
    """Budget-aware deterministic auction bidder.

    The live simulator may send one car at a time as a dictionary. This class
    loads the fitted model and fitted encoders once at initialization, then uses
    only stored preprocessing statistics during evaluation.
    """

    def __init__(
        self,
        bankroll: float = 500_000.0,
        model_path: str = "model_Daksh.pkl",
        encoder_path: str = "encoders_Daksh.pkl",
    ):
        self.bankroll = float(bankroll)
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.predicted_value = 0.0

        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)
        with self.encoder_path.open("rb") as f:
            self.encoders = pickle.load(f)

    def evaluate_car(self, car: Dict) -> float:
        """Predict the hammer price for a single car dictionary."""
        row = pd.DataFrame([car])
        features = self.encoders.transform(row)
        self.predicted_value = float(self.model.predict(features)[0])
        return self.predicted_value

    def place_bid(self, current_highest_bid: float, auction_round: int) -> float:
        """Return a deterministic bid from value, bankroll, high bid, and round.

        The bid ceiling starts conservative and rises by round, preserving a
        minimum expected margin while preventing a single car from consuming an
        excessive share of the remaining bankroll.
        """
        current_highest_bid = float(current_highest_bid)
        auction_round = max(1, int(auction_round))

        value = max(0.0, float(self.predicted_value))
        round_aggression = min(0.94, 0.70 + 0.04 * (auction_round - 1))
        value_ceiling = value * round_aggression
        bankroll_ceiling = self.bankroll * min(0.18, 0.07 + 0.015 * auction_round)
        max_bid = min(value_ceiling, bankroll_ceiling, self.bankroll)

        next_bid = current_highest_bid + max(100.0, 0.01 * value)
        if next_bid <= max_bid:
            return round(next_bid, 2)
        return 0.0

    def settle_auction(self, winning_bid: float) -> None:
        """Deduct a successful purchase from bankroll."""
        winning_bid = float(winning_bid)
        if winning_bid < 0:
            raise ValueError("winning_bid cannot be negative")
        if winning_bid > self.bankroll:
            raise ValueError("winning_bid exceeds bankroll")
        self.bankroll -= winning_bid


# Common simulator alias.
Agent = BiddingAgent
