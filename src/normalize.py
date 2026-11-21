"""Regex normalization of noisy chat text (technique area B).

Steps, in order:
  1. lowercase, unify curly quotes
  2. URLs -> "url", @mentions -> "@user", strip "#" from hashtags
  3. leetspeak inside words:  sh1t -> shit, b!tch -> bitch, $hit -> shit, f@ck -> f*ck
     ('@' inside a word becomes the '*' wildcard used by the edit-distance matcher)
  4. repeated letters: 3+ identical letters are cut to 2 (stuuuupid -> stuupid)
  5. collapse whitespace
"""
import re

_URL = re.compile(r"(https?://\S+|www\.\S+|\burl\b)", re.I)
_MENTION = re.compile(r"(?<![\w*$!@])@\w+")  # a mention starts at a word boundary
_HASHTAG = re.compile(r"#(\w+)")
_REPEAT = re.compile(r"([a-z])\1{2,}")
_SPACES = re.compile(r"\s+")
# A "word" that mixes letters with leet characters, e.g. "sh1t", "f@ck", "b!tch", "$hit".
_LEET_WORD = re.compile(r"(?<!\w)[a-z0-9@$!*]*[a-z][a-z0-9@$!*]*")
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "*", "$": "s", "!": "i"})
TOKEN_RE = re.compile(r"[a-z*]+(?:'[a-z]+)?")


def _deleet(match: re.Match) -> str:
    word = match.group(0)
    if word == "@user":
        return word
    # only translate words that contain both letters and leet symbols, and are not pure numbers
    if any(c in word for c in "013457@$!") and any(c.isalpha() for c in word):
        core = word.rstrip("!")  # keep sentence-final "!!!" as punctuation
        return core.translate(_LEET_MAP) + word[len(core):]
    return word


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = _URL.sub(" url ", t)
    t = _MENTION.sub(" @user ", t)
    t = _HASHTAG.sub(r"\1", t)
    t = _LEET_WORD.sub(_deleet, t)
    t = _REPEAT.sub(r"\1\1", t)
    return _SPACES.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    """Word tokens of an already normalized text (keeps '*' so masked words like f**k survive)."""
    return [tok for tok in TOKEN_RE.findall(text) if tok != "user"]


if __name__ == "__main__":
    for s in ["@JohnDoe You are SOOOOO stuuuupid!!! #MAGA http://t.co/x", "what a sh1t game, f@ck this b!tch", "u r an a$$hole lol, sh!t happens, f**k off"]:
        print(s, "->", normalize(s), tokenize(normalize(s)))
