"""The caption speaker: one line per caption, never blocks, fails soft. No audio is played:
a small Python process stands in for the speech engine and records what it receives."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from speech import CaptionSpeaker, clean  # noqa: E402


def _recorder(path):
    code = (f"import sys\nwith open({str(path)!r}, 'w', encoding='utf-8') as f:\n"
            "    for line in sys.stdin:\n        f.write(line); f.flush()\n")
    return [sys.executable, "-c", code]


def test_each_caption_becomes_one_clean_line(tmp_path):
    out = tmp_path / "said.txt"
    with CaptionSpeaker(_recorder(out)) as sp:
        assert sp.on
        t0 = time.perf_counter()
        for w in ["book", "drink\nnow", "thank\tyou"]:
            sp.say(w)
        assert time.perf_counter() - t0 < 0.5      # writing to the pipe does not wait for speech
    assert out.read_text(encoding="utf-8").splitlines() == ["book", "drinknow", "thankyou"]


def test_off_when_no_engine():
    sp = CaptionSpeaker(command=[])
    assert not sp.on and sp.reason_off
    sp.say("book")                                 # silently ignored
    sp.close()


def test_off_when_engine_cannot_start():
    sp = CaptionSpeaker(command=["definitely-not-a-real-program-echogesture"])
    assert not sp.on and "could not start" in sp.reason_off


def test_engine_dying_turns_speech_off(tmp_path):
    sp = CaptionSpeaker([sys.executable, "-c", "pass"])
    sp.proc.wait(timeout=10)
    sp.say("book")
    assert not sp.on
    sp.close()


def test_clean():
    assert clean("  hello\r\n") == "hello" and clean(42) == "42"
