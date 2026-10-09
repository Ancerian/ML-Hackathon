"""test_gs_gate.py — Unit tests for gs_gate module."""
import unittest
import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gs_gate import gate, DEFAULT_G_REF


class TestGSGate(unittest.TestCase):

    def test_gate_hard(self):
        # Score with low inconsistency should pass unchanged
        self.assertEqual(gate(None, 0.8, g_ref=0.634, mode="hard", precomputed_g=0.03), 0.8)
        # Score above threshold should be completely zeroed
        self.assertEqual(gate(None, 0.8, g_ref=0.634, mode="hard", precomputed_g=0.85), 0.0)
        # Boundary condition
        self.assertEqual(gate(None, 0.8, g_ref=0.634, mode="hard", precomputed_g=0.634), 0.0)

    def test_gate_sigmoid(self):
        # Exactly at g_ref, sigmoid multiplier is 0.5
        s_mid = gate(None, 1.0, g_ref=0.5, mode="sigmoid", k=10.0, precomputed_g=0.5)
        self.assertTrue(np.isclose(s_mid, 0.5))
        # Well below g_ref, multiplier near 1.0
        s_low = gate(None, 1.0, g_ref=0.5, mode="sigmoid", k=10.0, precomputed_g=0.1)
        self.assertGreater(s_low, 0.99)
        # Well above g_ref, multiplier near 0.0
        s_high = gate(None, 1.0, g_ref=0.5, mode="sigmoid", k=10.0, precomputed_g=0.9)
        self.assertLess(s_high, 0.01)

    def test_gate_log(self):
        # Below g_ref, no penalty
        s_low = gate(None, 0.7, g_ref=0.6, mode="log", beta=1.0, precomputed_g=0.4)
        self.assertTrue(np.isclose(s_low, 0.7))
        # Above g_ref, exponential decay
        s_high = gate(None, 0.7, g_ref=0.6, mode="log", beta=1.0, precomputed_g=1.2)
        expected = 0.7 * np.exp(-1.0 * (1.2 - 0.6) / 0.6)
        self.assertTrue(np.isclose(s_high, expected))

    def test_invalid_values(self):
        # NaN inconsistency should zero out score
        self.assertEqual(gate(None, 0.8, mode="hard", precomputed_g=np.nan), 0.0)


if __name__ == "__main__":
    unittest.main()

