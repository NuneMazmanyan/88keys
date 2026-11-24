import pandas as pd

from pianoscribe.data import splits as S


def test_maestro_and_asap_leakage(tmp_path):
    pd.DataFrame(
        {
            "split": ["train", "train", "test", "validation"],
            "audio_filename": ["2004/a.wav", "2004/b.wav", "2006/c.wav", "2008/d.wav"],
        }
    ).to_csv(tmp_path / "maestro.csv", index=False)
    maestro = S.maestro_splits(tmp_path / "maestro.csv")
    assert maestro == {"train": ["a", "b"], "validation": ["d"], "test": ["c"]}

    pd.DataFrame(
        {
            "midi_performance": [
                "Bach/Fugue1/p1.mid",  # train, piece also in test -> excluded
                "Bach/Fugue1/p2.mid",  # test
                "Chopin/Etude/p1.mid",  # train, clean
                "Liszt/X/p1.mid",  # not in MAESTRO -> ignored
            ],
            "maestro_audio_performance": [
                "{maestro}/2004/a.wav",
                "{maestro}/2006/c.wav",
                "{maestro}/2004/b.wav",
                None,
            ],
        }
    ).to_csv(tmp_path / "asap.csv", index=False)
    asap, excluded = S.asap_beat_splits(tmp_path / "asap.csv", maestro)
    assert asap["train"] == ["Chopin/Etude/p1.mid"]
    assert asap["test"] == ["Bach/Fugue1/p2.mid"]
    assert excluded == ["Bach/Fugue1"]


def test_song_split_is_deterministic_and_disjoint():
    ids = [f"{i:03d}" for i in range(1, 910)]
    a, b = S.song_level_split(ids), S.song_level_split(ids)
    assert a == b
    assert sum(len(v) for v in a.values()) == 909
    assert not set(a["train"]) & set(a["test"])


def test_hash_split_stable():
    assert S.hash_split("rooms/hall.wav") == S.hash_split("rooms/hall.wav")
