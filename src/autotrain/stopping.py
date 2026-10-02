import math


class EarlyStopping:
    def __init__(self, patience=3, threshold=0.0001, min_steps=10):
        if type(patience) is not int or patience <= 0:
            raise ValueError("Early stopping patience must be a positive integer")
        if type(min_steps) is not int or min_steps < 0:
            raise ValueError("Early stopping min_steps must be a nonnegative integer")
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError("Early stopping threshold must be finite and nonnegative")
        self.patience = patience
        self.threshold = threshold
        self.min_steps = min_steps
        self.best_loss = None
        self.best_step = None
        self.bad_checks = 0
        self.step = 0

    @property
    def stopped(self):
        return self.step >= self.min_steps and self.bad_checks >= self.patience

    def observe(self, step, loss):
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        improved = self.best_loss is None or loss < self.best_loss
        meaningful = self.best_loss is None or self.best_loss - loss > self.threshold
        self.bad_checks = 0 if meaningful else self.bad_checks + 1
        if improved:
            self.best_loss = loss
            self.best_step = step
        self.step = step
        return improved

    def state(self):
        return {
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "bad_checks": self.bad_checks,
            "step": self.step,
        }

    def restore(self, state):
        if (
            not math.isfinite(state["best_loss"])
            or not 0 < state["best_step"] <= state["step"]
            or state["bad_checks"] < 0
        ):
            raise ValueError("Invalid early stopping checkpoint state")
        self.best_loss = state["best_loss"]
        self.best_step = state["best_step"]
        self.bad_checks = state["bad_checks"]
        self.step = state["step"]
