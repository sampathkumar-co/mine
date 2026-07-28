from __future__ import annotations

import base64
import binascii
from typing import Callable, TypedDict

import pybase64


class Problem(TypedDict):
    plaintext: bytes


class Solution(TypedDict):
    encoded_data: bytes


def binascii_direct(problem: Problem) -> Solution:
    return {
        "encoded_data": binascii.b2a_base64(
            problem["plaintext"],
            newline=False,
        )
    }


def pybase64_simd(problem: Problem) -> Solution:
    return {"encoded_data": pybase64.b64encode(problem["plaintext"])}


def stdlib_control(problem: Problem) -> Solution:
    return {"encoded_data": base64.b64encode(problem["plaintext"])}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "binascii_direct": binascii_direct,
    "pybase64_simd": pybase64_simd,
    "stdlib_control": stdlib_control,
}
