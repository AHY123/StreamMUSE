# ...existing code...
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def _flatten_values(t: torch.Tensor, nbins: Optional[int] = None) -> np.ndarray:
    arr = t.cpu().numpy().ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    # continuous -> discretize into nbins
    if nbins is None:
        nbins = 256
    bins = np.linspace(arr.min(), arr.max(), nbins + 1)
    return np.digitize(arr, bins)


def value_counts(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, counts = np.unique(arr, return_counts=True)
    order = np.argsort(counts)[::-1]
    return vals[order], counts[order]


def group_by_rank_counts(counts: np.ndarray, group_size: int) -> np.ndarray:
    # counts is sorted desc; group into chunks of group_size
    chunks = [counts[i : i + group_size].sum() for i in range(0, len(counts), group_size)]
    return np.array(chunks)


def merge_small_counts(counts: np.ndarray, min_pct: float) -> np.ndarray:
    total = counts.sum()
    counts_list = list(counts)
    # merge smallest two until all groups >= min_pct
    while True:
        pct = np.array(counts_list) / total
        if pct.min() >= min_pct or len(counts_list) <= 1:
            break
        # merge two smallest
        idx_sorted = np.argsort(counts_list)
        i1, i2 = idx_sorted[0], idx_sorted[1]
        # ensure merge into single new group
        new = counts_list[i1] + counts_list[i2]
        # remove higher index first to avoid shifting
        for idx in sorted([i1, i2], reverse=True):
            counts_list.pop(idx)
        counts_list.append(new)
    # return descending-sorted array
    out = np.array(sorted(counts_list, reverse=True))
    return out


def cumulative_coverage(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = counts.sum()
    cum = np.cumsum(counts) / total
    x = np.arange(1, len(counts) + 1)
    return x, cum


def plot_cumulative_from_tensor(
    tensor: torch.Tensor,
    *,
    nbins_for_continuous: Optional[int] = None,
    group_n: Optional[int] = None,
    min_pct: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 4),
    savepath: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    tensor: 2D-or-more tensor where last dim is token (will flatten all tokens)
    group_n: if set, group every `group_n` most-frequent types into one group
    min_pct: if set, iteratively merge smallest groups until every group's pct >= min_pct
    If both group_n and min_pct are None, plot raw token-type cumulative coverage vs rank.
    """
    arr = _flatten_values(tensor, nbins_for_continuous)
    _, counts = value_counts(arr)
    # counts sorted desc
    if group_n is not None and group_n > 1:
        counts = group_by_rank_counts(counts, group_n)
    if min_pct is not None:
        counts = merge_small_counts(counts, min_pct)
    x, cum = cumulative_coverage(counts)

    plt.figure(figsize=figsize)
    plt.plot(x, cum * 100, marker="o")
    plt.xlabel("Number of groups / token types included")
    plt.ylabel("Cumulative coverage (%)")
    if title is None:
        title = "Cumulative coverage"
    plt.title(title)
    plt.grid(alpha=0.3)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage with your files
    mel = torch.load("data/pop909_mel_cp4.pt")
    acc = torch.load("data/pop909_acc_cp4.pt")
    print("mel.shape:", mel.shape)
    print("acc.shape:", acc.shape)

    # Treat last dim as tokens: flatten all tokens
    # Example A: raw per-type cumulative
    plot_cumulative_from_tensor(
        mel,
        nbins_for_continuous=None,
        group_n=None,
        min_pct=None,
        savepath="cum_raw_mel.png",
        title="mel cumulative (raw types)",
    )

    # Example B: group every 5 most-frequent types together
    plot_cumulative_from_tensor(mel, group_n=5, savepath="cum_group5_mel.png", title="mel cumulative (groups of 5)")

    # Example C: merge small groups until each group >= 5% coverage
    plot_cumulative_from_tensor(
        mel, min_pct=0.05, savepath="cum_merge5pct_mel.png", title="mel cumulative (merge small until >=5%)"
    )
# ...existing code...
