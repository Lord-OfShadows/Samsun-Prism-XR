"""Offline text-to-speech for the live captions.

The caption is spoken by the speech engine built into Windows (System.Speech, the SAPI
voices), so no package and no network is needed. One PowerShell process is started when
the demo starts and kept alive; each new caption is one line written to its stdin.

Why this shape:
- The video loop must never wait for audio. Writing a short line to a pipe returns at
  once, and the speaking happens in the other process.
- Captions can change faster than a word is spoken. Each new line cancels whatever is
  still being said, so the voice always says the current caption and never lags behind
  it with a queue of stale words.
- Starting PowerShell takes about a second, so it is done once, not once per word.

On a system without System.Speech (Linux, macOS) the speaker reports that it is off and
the demo runs silently.
"""
import subprocess
import sys

# Reads one caption per line and speaks it; a new line interrupts the previous one.
_PS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "while (($line = [Console]::In.ReadLine()) -ne $null) { "
    "$s.SpeakAsyncCancelAll(); if ($line.Trim()) { [void]$s.SpeakAsync($line) } }"
)


def default_command():
    """The command that starts the speech process, or None if this OS has no engine we use."""
    if sys.platform != "win32":
        return None
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_SCRIPT]


def clean(word):
    """One caption -> one safe line: no line breaks, no control characters."""
    return "".join(ch for ch in str(word) if ch.isprintable()).strip()


class CaptionSpeaker:
    """Speaks each new caption without blocking the caller.

    `command` is the process to start; it must read one line per caption from stdin.
    It is injectable so tests can run without a speech engine.
    """

    def __init__(self, command=None):
        command = default_command() if command is None else command
        self.proc, self.reason_off = None, None
        if not command:
            self.reason_off = "no offline speech engine on this OS (needs Windows System.Speech)"
            return
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        except OSError as e:
            self.reason_off = f"could not start the speech engine: {e}"

    @property
    def on(self):
        return self.proc is not None and self.proc.poll() is None

    def say(self, word):
        """Queue `word` to be spoken now, interrupting any earlier word. Never blocks on audio."""
        if not self.on:
            return
        try:
            self.proc.stdin.write(clean(word) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.proc = None
            self.reason_off = "the speech engine stopped"

    def close(self, timeout=2.0):
        """Stop speaking and end the process."""
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
