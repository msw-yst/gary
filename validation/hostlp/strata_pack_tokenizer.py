"""strata_pack_tokenizer.py - tokadapt module adapter for a Strata engine pack (DeepSeek V4.1 Flash run, 2026-11-16).

The pack has no Hugging Face tokenizer.json (its tokenizer.json is Strata's small config), so the serving stack's own
tokenizer class is used, built exactly as the server builds it: strata_tokenizer.Tokenizer from
pack/tokenizer/{vocab.json, merges.txt, token_type.json} with the pre-tokenizer named in the config and the special ids,
encoding with parse_special=True so "</think>" and friends map to their special ids.

Env:  STRATA_SRV  the server source tree (holding tools/strata_tokenizer.py)
      STRATA_PYDEPS  its Python deps (the original run needed Python 3.12 for the compiled `regex` wheel)
      STRATA_PACK  the pack dir (holding tokenizer/)

    STRATA_SRV=... STRATA_PYDEPS=... STRATA_PACK=... python3 map_tokens.py --tokenizer module:strata_pack_tokenizer.py ...
"""
import json
import os
import sys


class _T:
    def __init__(self):
        srv, deps, pack = os.environ["STRATA_SRV"], os.environ.get("STRATA_PYDEPS", ""), os.environ["STRATA_PACK"]
        sys.path[:0] = [p for p in (os.path.join(srv, "tools"), deps) if p]
        import strata_tokenizer as ST
        tp = os.path.join(pack, "tokenizer")
        vocab = json.load(open(os.path.join(tp, "vocab.json"), encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        cfg = json.load(open(os.path.join(tp, "tokenizer.json")))
        self.types = json.load(open(os.path.join(tp, "token_type.json")))
        merges = open(os.path.join(tp, "merges.txt"), encoding="utf-8").read().split("\n")
        self.t = ST.Tokenizer(tokens, merges, self.types, cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)
        self.n_vocab = len(tokens)

    def encode(self, text):
        return self.t.encode(text, parse_special=True)

    def token_bytes(self, i):
        return self.t.token_bytes(i)

    def is_normal(self, i):
        return self.types[i] == 1


def load():
    return _T()
