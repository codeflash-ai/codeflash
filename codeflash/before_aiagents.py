def _estimate_string_tokens(content: str | Sequence[UserContent]) -> int:
    if not content:
        return 0
    if isinstance(content, str):
        return len(re.split(r'[\s",.:]+', content.strip()))
    tokens = 0
    for part in content:
        if isinstance(part, str):
            tokens += len(re.split(r'[\s",.:]+', part.strip()))
        if isinstance(part, (AudioUrl, ImageUrl)):
            tokens += 0
        elif isinstance(part, BinaryContent):
            tokens += len(part.data)
        else:
            tokens += 0
    return tokens
