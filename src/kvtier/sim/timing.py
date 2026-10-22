"""Step-time model on top of the C3 cost model."""

from kvtier.cost import CostModel


class StepModel:
    def __init__(self, cost: CostModel):
        self.cost = cost
        self.cal = cost.cal

    def prefill_s(self, start: int, n: int) -> float:
        """n new tokens after `start` cached ones: f(start + n) - f(start) on the cold-prefill curve."""
        if n <= 0:
            return 0.0
        return self.cost.recompute_s(start + n) - self.cost.recompute_s(start)

    def decode_s(self, n_running: int) -> float:
        return 0.0 if n_running == 0 else self.cal.decode_base_s + self.cal.decode_per_request_s * n_running

    def reload_s(self, tokens: int) -> float:
        return 0.0 if tokens <= 0 else self.cost.reload_s(tokens)
