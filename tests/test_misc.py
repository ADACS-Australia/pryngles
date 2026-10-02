"""Tests for the functions in the ``misc`` module."""

from unittest import mock

import pandas as pd
import pytest

import pryngles as pr
from pryngles.consts import ROOTDIR
from pryngles.misc import get_data, flatten, get_methods, calc_hash, retrieve_data


def test_get_data():
    """``get_data`` returns the full path to a packaged data file."""
    path = get_data("diffuse_reflection_function.data")
    assert path == ROOTDIR + "/data/diffuse_reflection_function.data"


def test_flatten():
    """``flatten`` recursively flattens nested iterables but keeps strings atomic."""
    assert list(flatten(["hola"])) == ["hola"]
    assert list(flatten(["hola", ["perro", "gato"]])) == [
        "hola",
        "perro",
        "gato",
    ]
    assert list(flatten([[1, "perro"], object, 2.5])) == [
        1,
        "perro",
        object,
        2.5,
    ]


class _SampleClass:
    """A sample class with a mix of public and private methods."""

    def public_method(self):
        pass

    def another_public(self):
        pass

    def _private_method(self):
        pass

    def __dunder_method__(self):
        pass


def test_get_methods():
    """``get_methods`` returns the sorted public methods of a class."""
    methods = get_methods(_SampleClass)
    assert methods == sorted(methods)
    assert methods == ["_private_method", "another_public", "public_method"]


def test_calc_hash_dict():
    """``calc_hash`` is deterministic for a given dict."""
    d = dict(a=1, b=3, c=pd)
    assert calc_hash(d) == calc_hash(d)


def test_calc_hash_object():
    """``calc_hash`` works on arbitrary objects and classes."""
    # Hash is not deterministic on objects, but it should return a string.
    assert isinstance(calc_hash(_SampleClass()), str)
    assert isinstance(calc_hash(_SampleClass), str)


@mock.patch("pryngles.misc.gdown.download")
@mock.patch("pryngles.misc.pd.read_csv")
def test_retrieve_data(mock_read, mock_download, tmp_path):
    """``retrieve_data`` downloads the requested files from the data index."""
    # A fake data index mapping filenames to Google Drive file ids.
    mock_read.return_value = pd.DataFrame(
        {"fileid": ["abc123", "def456"]},
        index=["star.dat", "planet.dat"],
    )

    files = retrieve_data(["star.dat", "planet.dat"], path=str(tmp_path), quiet=True)

    # The index file is downloaded once, plus one download per requested file.
    assert mock_read.call_count == 1
    assert mock_download.call_count == 3
    assert files == [str(tmp_path / "star.dat"), str(tmp_path / "planet.dat")]


@mock.patch("pryngles.misc.gdown.download")
@mock.patch("pryngles.misc.pd.read_csv")
def test_retrieve_data_missing(mock_read, mock_download, tmp_path):
    """``retrieve_data`` raises for a file not present in the index."""
    index = pd.DataFrame({"fileid": ["abc123"]}, index=["star.dat"])
    mock_read.return_value = index

    with pytest.raises(ValueError, match="not available"):
        retrieve_data("missing.dat", path=str(tmp_path), quiet=True)
