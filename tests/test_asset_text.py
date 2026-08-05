from pathlib import Path


ASSET = Path(__file__).resolve().parents[1] / "source" / "dicedial" / "assets" / "numbered_die.usda"


def test_numbered_die_asset_has_six_faces_and_twenty_one_pips():
    text = ASSET.read_text()
    assert text.startswith("#usda 1.0")
    for face in range(1, 7):
        assert f'def Xform "Face{face}"' in text
    assert text.count('def Sphere "Pip') == 21
    assert text.count('def Cube "Collision"') == 1


def test_numbered_die_asset_has_balanced_braces():
    text = ASSET.read_text()
    assert text.count("{") == text.count("}")
