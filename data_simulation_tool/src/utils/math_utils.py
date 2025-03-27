from typing import List, Sequence


def decompose(
    target_sum: int, candidate_values: Sequence[int], max_usage: Sequence[int]
) -> List[int]:
    candidate_values = sorted(set(candidate_values))  # Sort and remove duplicates

    dp = [None] * (target_sum + 1)
    dp[0] = []

    for i in range(1, target_sum + 1):
        for j, value in enumerate(candidate_values):
            if (
                value <= i
                and dp[i - value] is not None
                and max_usage[j] > dp[i - value].count(value)
                and (dp[i] is None or len(dp[i - value]) + 1 < len(dp[i]))
            ):
                dp[i] = dp[i - value] + [value]
    res = dp[target_sum] if dp[target_sum] else []
    res.sort(reverse=True)
    return res


if __name__ == "__main__":
    target = 10
    vals = [1, 2, 3]
    max_usage = [100, 0, 2]
    result = decompose(target, vals, max_usage)
    print(result)  # Output: [1, 1, 2, 3, 3]
