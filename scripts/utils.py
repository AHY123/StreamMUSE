def split_at_nth_underscore(text: str, n: int) -> tuple[str, str] | None:
    """
    在字符串的第 N 个下划线 (_) 处进行分割。

    Args:
        text: 原始字符串。
        n: 分割点是第 N 个下划线 (例如 N=2 表示第二个下划线)。

    Returns:
        包含两个部分的元组 (part1, part2)，如果下划线数量不足则返回 None。
    """
    # 1. 完整地按所有下划线分割
    parts = text.split("_")

    # 2. 检查是否有足够的下划线 (即 parts 数量是否足够)
    # n=2 需要至少 n+1 = 3 个部分 (part[0], part[1], part[2]...)
    if len(parts) < n + 1:
        return None

    # 3. 重新组合：第一部分是 parts[0] 到 parts[n-1]，用 '_' 连接
    part1_list = parts[:n]
    part1 = "_".join(part1_list)

    # 4. 重新组合：第二部分是从 parts[n] 开始到结尾，用 '_' 连接
    part2_list = parts[n:]
    part2 = "_".join(part2_list)

    return (part1, part2)


def find_unique_path_with_target(paths: list[str], target: str) -> str | None:
    """
    在一组路径中查找包含特定 'target' 文本的唯一路径。

    Args:
        paths: 包含数百个路径字符串的列表。
        target: 要查找的子串（保证只在一个路径中出现）。

    Returns:
        匹配的路径字符串，如果找不到任何匹配项则返回 None。
    """
    try:
        # 使用生成器表达式 (path for path in paths if target in path)
        # next() 函数会返回第一个匹配的 path，并在找到后立即停止遍历。
        return next(path for path in paths if target in path)
    except StopIteration:
        # 如果 next() 遍历完所有元素仍未找到匹配项，会抛出 StopIteration 异常。
        return None
