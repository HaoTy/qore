import numpy as np


dp = []

def check_cone_validity(bitstr: str) -> bool:
    assert (len(bitstr)**0.5 + 0.5)**2 == len(bitstr), "Length of the bitstring should be a perfect square."
    if not dp:
        compute_validity_dp(len(bitstr))
    return dp[len(bitstr)-1][int(bitstr, 2)]


def check_cone_validity_aux(bitstr: int, n: int) -> bool:
    if n == 1:
        return True
    if not dp[n-2][bitstr >> n**2 - (n - 1)**2]:
        return False
    for i in range((n - 2)**2, (n - 1)**2):
        if (bitstr & (1 << i)) & ~(bitstr & (111 << (i - (n - 2)**2 + (n - 1)**2))):
            return False
    return True
    

def compute_validity_dp(n: int) -> None:
    for i in range(n):
        dp.append(np.zeros(2**((i+1)**2), dtype=bool))
        for j in range(2**((i+1)**2)):
            dp[i][j] = check_cone_validity_aux(j, i)
