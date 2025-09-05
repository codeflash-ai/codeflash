def _estimate_string_tokens(content: str | Sequence[UserContent]) -> int:
    if not content:
        return 0
    if isinstance(content, str):
        return len(_TOKEN_SPLIT_RE.split(content.strip()))
    tokens = 0
    for part in content:
        if isinstance(part, str):
            tokens += len(_TOKEN_SPLIT_RE.split(part.strip()))
        elif isinstance(part, BinaryContent):
            tokens += len(part.data)
    return tokens


_TOKEN_SPLIT_RE = re.compile(r'[\s",.:]+')
