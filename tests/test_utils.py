import pytest 
import tempfile 

from pathlib import Path 
from recsys import utils 

def test_save_and_load_dict():
    with tempfile.TemporaryDirectory() as dp:
        d = {"hello":"world"}
        fp = Path(dp, "d.json")
        utils.save_dict(d=d, filepath=fp)
        d = utils.load_dict(filepath=fp)
        assert d["hello"] == "world"
