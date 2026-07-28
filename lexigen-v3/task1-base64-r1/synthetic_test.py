from __future__ import annotations

import base64
import os

from candidates import CANDIDATES


for size in (0, 1, 2, 3, 4, 5, 63, 64, 65, 1024, 1_000_003):
    plaintext = os.urandom(size)
    expected = base64.b64encode(plaintext)
    problem = {"plaintext": plaintext}
    for name, candidate in CANDIDATES.items():
        result = candidate(problem)
        assert result["encoded_data"] == expected, (size, name)
        assert isinstance(result["encoded_data"], bytes), (size, name)

print("synthetic Base64 exactness passed")
