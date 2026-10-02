import unittest

from autotrain.stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_plateau_and_small_improvements_exhaust_patience(self):
        stopping = EarlyStopping(patience=3, threshold=0.001, min_steps=10)
        for step, loss in [(5, 0.5), (10, 0.4999), (15, 0.4998), (20, 0.4997)]:
            self.assertTrue(stopping.observe(step, loss))
        self.assertTrue(stopping.stopped)
        self.assertEqual(stopping.best_step, 20)

    def test_meaningful_improvement_resets_patience(self):
        stopping = EarlyStopping(patience=2, min_steps=10)
        for step, loss in [(5, 1.0), (10, 1.0), (15, 0.5), (20, 0.5)]:
            stopping.observe(step, loss)
            self.assertFalse(stopping.stopped)
        stopping.observe(25, 0.5)
        self.assertTrue(stopping.stopped)

    def test_min_steps_and_invalid_settings(self):
        stopping = EarlyStopping(patience=1, min_steps=10)
        stopping.observe(1, 1.0)
        stopping.observe(2, 2.0)
        self.assertFalse(stopping.stopped)
        for settings in [
            {"patience": 0},
            {"patience": True},
            {"min_steps": -1},
            {"threshold": float("nan")},
            {"threshold": -1},
        ]:
            with self.assertRaises(ValueError):
                EarlyStopping(**settings)
        with self.assertRaises(ValueError):
            stopping.observe(10, float("inf"))
