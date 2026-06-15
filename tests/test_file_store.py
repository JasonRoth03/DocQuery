import pytest
from app.services.file_store import LocalFileStore


@pytest.fixture
def store(tmp_path):
    return LocalFileStore(base_dir=tmp_path)


async def test_save_and_load(store):
    data = b"hello file store"
    key = await store.save(data, "hello.txt")
    assert await store.load(key) == data


async def test_save_returns_unique_keys_for_same_filename(store):
    data = b"same content"
    key1 = await store.save(data, "file.txt")
    key2 = await store.save(data, "file.txt")
    assert key1 != key2


async def test_key_contains_original_filename(store):
    key = await store.save(b"data", "myreport.pdf")
    assert "myreport.pdf" in key


async def test_delete_removes_file(store, tmp_path):
    key = await store.save(b"delete me", "temp.txt")
    await store.delete(key)
    assert not (tmp_path / key).exists()


async def test_delete_nonexistent_is_noop(store):
    await store.delete("nonexistent/ghost.txt")  # must not raise


async def test_load_missing_file_raises(store):
    with pytest.raises(FileNotFoundError):
        await store.load("ghost/missing.txt")


async def test_multiple_files_stored_independently(store):
    keys = []
    for i in range(5):
        key = await store.save(f"content {i}".encode(), f"file{i}.txt")
        keys.append(key)

    for i, key in enumerate(keys):
        assert await store.load(key) == f"content {i}".encode()
