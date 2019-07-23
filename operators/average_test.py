from unittest import TestCase

import numpy as np

from matplotlib import pyplot as plt

from debug_tools import error_tracking_tools
import operators.average
from debug_tools.visualization import WindowManager


class TestDiff_backward_e1(TestCase):
    def test_average_backward_e1(self):
        arr = np.ones(100)
        arr[::2] = 0
        res = operators.average.avg_backward_e1(arr)
        self.assertEqual(res.min(), res.max())
        self.assertEqual(res.min(), 0.5)

class TestDiff_forward_e1(TestCase):
    def test_average_forward_e1(self):
        arr = np.ones(100)
        arr[::2] = 0
        res = operators.average.avg_forward_e1(arr)
        self.assertEqual(res.min(), res.max())
        self.assertEqual(res.min(), 0.5)

class TestDiff_e2(TestCase):
    def test_average_e2(self):
        arr = np.ones(100)
        arr[::2] = 0
        res = operators.average.avg_e2(arr)
        self.assertEqual(res.min(), res.max())
        self.assertEqual(res.min(), 0.5)
