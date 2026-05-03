import sys
import os

sys.path.append(os.path.abspath("."))

from src.utils.metrics import masking_ratio


def test_masking():
    assert masking_ratio(2, [1, 1]) == 1.0