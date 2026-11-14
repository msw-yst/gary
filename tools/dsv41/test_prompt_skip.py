"""The long-prompt layer skip (STRATA_DS4_PROMPT_SKIP=1) against the full work: same answers within noise.

Needs a GPU, a DeepSeek-V4.1 pack and the golden ids file, so it runs only when they are named:

    STRATA_DSV41_BIN=.../strata-dsv41 STRATA_DSV41_PACK=/data/pack2.00 STRATA_DSV41_IDS=.../logpos.ids.json \
        python -m unittest tools.dsv41.test_prompt_skip

The golden segments are joined, the first DS4_SCORE_PREFIX tokens (4200) are read as one prompt (the path a long prompt
takes: logits of the last row only, where the skip applies), and the positions after them are scored teacher-forced.
The skip changes only what the layers 20-39 compute for the prompt's early rows, so the scored rows' KL against the
full work must stay at the noise of the experts' summing order.

A tensor-parallel group (STRATA_DS4_PROMPT_SKIP_TP=1): STRATA_DSV41_ARGS replaces the one-GPU arguments
("--tp 1 --threads off"), e.g. "--gpus 0,1,2 --tp 3 --helper-device 3,4,5 --profile-readonly"; when it holds --tp N with
N > 1 both arms run with STRATA_DS4_PROMPT_SKIP_TP=1 (the skip arm takes it, the full arm is STRATA_DS4_PROMPT_SKIP=0).
STRATA_DSV41_PREFIX (default 4200), STRATA_DSV41_CTX (default 6144) and STRATA_DSV41_WINDOW (default 128; 1000 scores the
~800 positions in one feed, one expert stream instead of seven) set the prompt, the context and the scored feeds.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import quant_quality  # noqa: E402

BIN = os.environ.get("STRATA_DSV41_BIN")
PACK = os.environ.get("STRATA_DSV41_PACK")
IDS = os.environ.get("STRATA_DSV41_IDS")
ARGS = os.environ.get("STRATA_DSV41_ARGS", "--tp 1 --threads off").split()
PREFIX = os.environ.get("STRATA_DSV41_PREFIX", "4200")
CTX = os.environ.get("STRATA_DSV41_CTX", "6144")
WINDOW = os.environ.get("STRATA_DSV41_WINDOW", "128")


def tp_of(args):
    """The --tp value of an argument list (1 when none)."""
    for i, a in enumerate(args[:-1]):
        if a == "--tp":
            return int(args[i + 1])
    return 1


def keep_rows(nl=40, yoco=20, win=128):
    """The rows a long prompt keeps on layers yoco.. (the engine's forward_chunks): each layer looks back one window
    more than the layer before, so the last (nl - yoco) * (win - 1) + win rows."""
    return (nl - yoco) * (win - 1) + win


class PromptSkip(unittest.TestCase):
    def test_keep_rows_is_the_chain_of_windows(self):
        self.assertEqual(keep_rows(), 2668)

    def test_tp_of(self):
        self.assertEqual(tp_of(["--tp", "1", "--threads", "off"]), 1)
        self.assertEqual(tp_of(["--gpus", "0,1,2", "--tp", "3"]), 3)
        self.assertEqual(tp_of([]), 1)

    @unittest.skipUnless(BIN and PACK and IDS, "needs STRATA_DSV41_BIN / _PACK / _IDS")
    def test_scored_rows_after_a_skipped_prompt_match_the_full_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            outs = {}
            for mode in ("0", "1"):
                out = pathlib.Path(tmp) / f"skip{mode}.tsv"
                env = dict(os.environ, STRATA_DS4_PROMPT_SKIP=mode, DS4_SCORE_PREFIX=PREFIX, DS4_CHUNK="1024")
                if tp_of(ARGS) > 1:
                    env["STRATA_DS4_PROMPT_SKIP_TP"] = "1"
                subprocess.run([BIN, "--pack", PACK, *ARGS, "--max-context", CTX, "--vision",
                                "off", "--dspark", "off", "--expert-profile", "none", "--window", WINDOW, "--score", IDS,
                                "--score-out", str(out)], check=True, env=env)
                outs[mode] = quant_quality.read_run(str(out))
            res = quant_quality.compare(outs["0"], outs["1"])["all"]
            self.assertGreater(res["positions"], 500)
            self.assertLess(res["kl"], 0.002)
            self.assertGreaterEqual(res["top1"], 0.99)


if __name__ == "__main__":
    unittest.main()
