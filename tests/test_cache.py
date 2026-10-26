from ss_taker.cache import CacheEntry, ResultCache


def test_cache_hit_requires_same_hash() -> None:
    cache = ResultCache()
    cache.set(CacheEntry(image_hash="abc", text="hello", output_path="x.txt"))
    assert cache.get("abc").text == "hello"
    assert cache.get("def") is None

