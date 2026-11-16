"""tokadapt.py - one tokenizer interface for HOSTLP's byte mapping, whatever the model.

load(spec) returns an object with:
    encode(text) -> [ids]          special tokens written in the text (e.g. "</think>") map to their special ids
    token_bytes(id) -> bytes       the exact bytes this id stands for
    n_vocab -> int
    is_normal(id) -> bool          False for control / special ids (used to break byte collisions in favour of normal ids)

Specs:
    hf:/path/tokenizer.json         a Hugging Face `tokenizers` file (pip install tokenizers). Byte-level BPE vocabularies
                                    are decoded through the GPT-2 byte table, so token_bytes is exact, not a decode() guess.
    module:/path/adapter.py         any Python file defining load() -> such an object (see strata_pack_tokenizer.py for the DeepSeek V4.1
                                    pack tokenizer that the original run used).

The check that matters is done by the caller (map_tokens.py): the hosted steps' bytes must re-encode to exactly one of
our ids per step, with byte-identical pieces. If that fails, the tokenizers are not the same and HOSTLP cannot be
teacher-forced; a byte-level fallback is needed (not published here).
"""
import importlib.util
import json


def _gpt2_byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class HF:
    def __init__(self, path):
        from tokenizers import Tokenizer  # imported here so module: specs need no tokenizers install
        self.t = Tokenizer.from_file(path)
        cfg = json.load(open(path, encoding="utf-8"))
        self.vocab = self.t.get_vocab(with_added_tokens=True)
        self.n_vocab = max(self.vocab.values()) + 1
        self.inv = {i: s for s, i in self.vocab.items()}
        self.special = {a["id"] for a in cfg.get("added_tokens", []) if a.get("special")}
        self.added = {a["id"]: a["content"] for a in cfg.get("added_tokens", [])}
        pre = json.dumps(cfg.get("pre_tokenizer") or {}) + json.dumps(cfg.get("decoder") or {})
        self.bytelevel = "ByteLevel" in pre
        self.bd = _gpt2_byte_decoder()

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False).ids

    def token_bytes(self, i):
        if i in self.added:
            return self.added[i].encode("utf-8")
        s = self.inv.get(i)
        if s is None:
            return b""
        if self.bytelevel:
            try:
                return bytes(self.bd[ch] for ch in s)
            except KeyError:
                pass
        if s.startswith("<0x") and s.endswith(">") and len(s) == 6:   # sentencepiece byte fallback
            return bytes([int(s[3:5], 16)])
        return self.t.decode([i], skip_special_tokens=False).encode("utf-8")

    def is_normal(self, i):
        return i not in self.special


def load(spec):
    kind, _, path = spec.partition(":")
    if kind == "hf":
        return HF(path)
    if kind == "module":
        sp = importlib.util.spec_from_file_location("tokadapt_ext", path)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m.load()
    raise SystemExit("tokenizer spec %r: want hf:FILE or module:FILE" % spec)
