from pathlib import PurePosixPath, PureWindowsPath

from blink import paths


def test_paths_default_to_blink_home_on_d():
    assert paths.default_home(platform="win32", env={}) == PureWindowsPath(r"D:\blink")


def test_blink_home_env_overrides_the_default():
    home = paths.default_home(platform="win32", env={"BLINK_HOME": r"E:\elsewhere"})
    assert home == PureWindowsPath(r"E:\elsewhere")


def test_linux_default_is_under_the_user_home():
    assert paths.default_home(platform="linux", env={"HOME": "/home/u"}) == PurePosixPath("/home/u/.blink")


def test_layout_names_every_heavy_state_directory():
    layout = paths.layout(PureWindowsPath(r"D:\blink"))
    assert set(layout) == {
        "data",
        "runs",
        "games",
        "eval",
        "ship",
        "export",
        "film",
        "logs",
        "dm",
        "books",
        "lichess",
        "downloads",
    }
    assert layout["runs"] == PureWindowsPath(r"D:\blink\runs")


def test_layout_is_read_only():
    layout = paths.layout(PureWindowsPath(r"D:\blink"))
    try:
        layout["runs"] = PureWindowsPath(r"C:\oops")  # type: ignore[index]
    except TypeError:
        return
    raise AssertionError("layout must be immutable")
