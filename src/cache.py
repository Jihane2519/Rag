from src.models import MinimalSource
_query_cache: dict[tuple[str, int], list[MinimalSource]] = {}


def clear_query_cache() -> None:
    """Clear cached query results."""
    _query_cache.clear()
