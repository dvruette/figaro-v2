from typing import Tuple

def normalize_time_sig(time_sig: Tuple[int, int]) -> Tuple[int, int]:
    num, denom = time_sig

    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    if denom == 0:
        return num, denom
    divisor = gcd(num, denom)
    num, denom = num // divisor, denom // divisor
    if denom == 1:
        num, denom = 4*num, 4*denom
    elif denom == 2:
        num, denom = 2*num, 2*denom
    return num, denom


def parse_time_sig(time_sig: str) -> Tuple[int, int]:
    num, denom = time_sig.split("/")
    return normalize_time_sig((int(num), int(denom)))
