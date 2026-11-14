// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.hpp - DeepSeek-V4.1-Flash on Maya: an EXL3 model (tools/ds4/pack.py's pack beside its files) over
// the machine's GPUs, layer-split, with the routed experts that do not fit in VRAM on the CPU lane.
//
// Windows of up to 16 tokens through every layer (a prompt in windows, a decode step a window of one): each query of a
// window sees exactly what the reference's decode path gives it (DeepSeek's inference/model.py at start_pos > 0) -
// its own 128-token window, the compressed groups completed by its position.  The state a token leaves is the
// 128-token window ring of every layer, the compressors' partial groups, the compressed KV and index keys of the KV
// source layers (append-only), and the Engram n-gram history; checkpoint() / restore() keep the first two (the rest is
// truncated by position) so a conversation can go back to the end of its last prompt.  A conversation that leaves the
// model altogether (another chat answered in between) is a slot: slot_save / slot_load take the same state - the
// compressed KV and index keys included, which a checkpoint leaves in place - through a file or a host buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4 {

struct Options {
    std::string pack;                 // <model>/maya-ds4
    int64_t max_context = 32768;
    std::vector<int> gpus;            // empty: every visible GPU
    int64_t reserve_mib = 600;        // VRAM left free on each GPU
    bool vram_experts_auto = true;    // (Metal: no --vram-experts given: the share the anonymous-memory cap leaves - metal_expert_cap)
    double vram_experts = 1.0;        // the share of the expert budget to use (tests: less, so the CPU lane works)
    int cpu_threads = 0;              // 0: the physical cores
    std::string profile;              // expert hit counts (read to place the hottest experts in VRAM, written back)
    // --profile-readonly: read the profile to place the experts, never write it back.  Without this the run's routing
    // counts go into the file, so the next run places experts differently and greedy output / KL move between runs.
    bool profile_readonly = false;
    bool expert_cache = true;         // experts move between VRAM and the CPU lane as they are used
    // the CPU lane's experts read in place from the model's files, the page cache keeping what fits, instead of
    // copied into RAM: 1 on, 0 off, -1 (auto) where the RAM this process can get does not hold them
    int mmap_experts = -1;
    // the experts the VRAM leaves over are cached in RAM by a store of their own (ds4_store.hpp): whole experts read unbuffered into
    // slots, a replacement policy (0 lru, 1 lfu, 2 lfu then lru), filled on demand and by the lookahead's guesses.  GB of RAM at
    // most (never past the RAM the process can still get less 8 GB); 0: off, the OS page cache as before
    double ssd_cache_gb = -1.0;   // < 0: auto (the RAM the process can get less a floor, at most 48 GB; off below 4 GB), 0: off
    int ssd_policy = 0;
    // dspark_auto: the drafter is loaded only where the routed experts fit the GPUs (more than 1.5x what they have, the verify rounds
    // read 3.3x the bytes of a token and measured no faster than plain decoding, and the drafter's 7 GB of experts take the VRAM the
    // trunk's hottest would have had); --dspark on / off / none says it outright
    // experts a word takes: -1 the rule (5 where the cards hold 16 GiB or less in all, else the model's 6), N the router's top N with
    // their weights renormalised (--experts 6: full quality); ds4_experts_rule.hpp
    int experts = -1;
    bool dspark_auto = true;
    bool dspark = true;              // load the DSpark drafter (the checkpoint's mtp.* blocks) for speculative decoding
    int dspark_block = 0;             // at most this many drafts a round (--dspark-block N; 0: the pack's block size)
    float dspark_conf = 0.5f;         // a draft is verified while its confidence (sigmoid) is at least this
    // DS4_DSPARK_NOSEED=1 (SV-09): the drafter is loaded (--dspark off) but this process never drafts - no
    // seeding after a window, no taps in layers 37..39, and draft() offers nothing.  ds4_main.cpp sets it only for a
    // --dspark off run that cannot ask for a draft (no --batch-spec, no --spec-eval); a serve request's spec=1 then gets
    // plain steps.  Outputs bit-identical (nothing the trunk computes reads the drafter's state)
    bool dspark_noseed = false;
    // tensor parallelism: runs of tp GPUs of one kind (the same memory) take the same layers, each a share of the
    // heads, of the output projection and of every expert's FFN, summed through host-mapped memory (a GPU unlike the
    // next stays alone); tp > 1 runs with threads (--tp 1 --threads off: one host thread, layer split only)
    int tp = 2;
    bool threads = false;             // a host thread per GPU enqueues its layers (the hand-offs host-mapped)
    // prompts in chunks of this many tokens through each layer at once (threads): the GEMMs on the tensor cores, the
    // CPU lane's experts streamed to the GPUs a chunk at a time instead of computed on the host; 0: windows of 16;
    // -1: 2048, or 8192 with the experts read in place (every chunk streams them all from the files)
    int prefill_chunk = -1;
    // prototype (ds4_helper.hpp): a GPU outside --gpus that owns no layers and holds the hottest routed experts the
    // primary's VRAM does not (--helper-device N / STRATA_DS4_HELPER=N; -1 off).  helper_slots 0: as many as fit.
    std::vector<int> helper_devices;   // --helper-device 1,4 / STRATA_DS4_HELPER: several helpers (empty: off); each layer's on one of them (DS4_HELPER_PLACE=stripe: striped in rank order)
    int64_t helper_slots = 0;
    bool vision = true;               // the vision tower (the checkpoint's vision.* / aligner.*), when it has one
    bool vision_only = false;         // (tests) load the vision tower alone: encode_image() and nothing else
    bool verbose = false;
};

/// A picture in the prompt: its span's rows (every position of the span carries the image token), BF16 [span][dim]
struct Image {
    int64_t start = 0;
    std::shared_ptr<const std::vector<uint16_t>> rows;
};

struct Stats {
    int64_t tokens = 0;
    int64_t experts = 0, cpu_experts = 0;   // routed experts run, and how many of them on the CPU lane
    int64_t helper_experts = 0;             // (ds4_helper.hpp) of them computed by the helper expert cache
    double helper_ms = 0.0;                 // the lane's host time issuing to and waiting for the helper
    double helper_arm_ms = 0.0;             // DS4_HELPER_ARMED: the lane's host time arming the next windows (after its answers)
    double cpu_ms = 0.0, gpu_wait_ms = 0.0;   // the CPU lane's time; the host's waits for the routing
    double layer_ms = 0.0, head_ms = 0.0, engram_ms = 0.0;   // the layers' host time in all; the head's wait; Engram reads
    int64_t eg_rows = 0, eg_hits = 0, eg_reads = 0;   // Engram rows asked for, served without a read, read from the files
    double eg_io_ms = 0.0;                            // the windows' Engram reads, issue to last done (mostly hidden)
    int64_t eg_cached = 0, eg_cache_cap = 0;          // the row cache's rows held, its capacity
    int64_t swaps = 0;                // experts the cache moved into VRAM
    int64_t skipped = 0;              // routed experts STRATA_DS4_DISK_SKIP_W left out (light, not in RAM)
    int64_t ram_experts = 0, ram_moves = 0;   // CPU-lane experts read from the RAM tier; experts moved into it
    double prof_ms[6] = {};          // DS4_PROF: the GPU's time in decode by stage (between layers, attention, routing +
    int64_t prof_tokens = 0;         // shared expert, experts' gate / up, the wait for the CPU lane, down + post)
    // lookahead (decode windows of one): the picks of layers guessed a layer before (the next layer's gate on this
    // layer's FFN input), of them in the guess's top k / in all its guesses / in the guess two layers before; the picks
    // read from the files, of them prefetched early; experts prefetched early
    int64_t ahead_picks = 0, ahead_hit_k = 0, ahead_hit = 0, ahead_hit2 = 0;
    int64_t ahead_cold = 0, ahead_cold_hit = 0, ahead_pf = 0;
    int64_t ahead_rank_n[16] = {}, ahead_rank_hit[16] = {};   // the next-layer guesses by rank, and how many were picked
    // the RAM store (--ssd-cache), totals since the load: reads asked for and found there (hits), missed (misses), started ahead
    // of use (prefetched) of which used / given up unused, slots given up, bytes read, ms the CPU lane waited for the reads
    int64_t store_hits = 0, store_misses = 0, store_prefetched = 0, store_pf_used = 0, store_pf_wasted = 0, store_evictions = 0;
    int64_t store_bytes = 0, store_fails = 0;
    int64_t store_zc = 0, store_zc_visible = 0;   // experts the GPU computed from the store (zero-copy over PCIe); slots it may read
    double store_wait_ms = 0.0, store_busy_ms = 0.0;
    int64_t store_slots = 0, store_slots_max = 0;
    int64_t drafts_offered = 0, drafts_accepted = 0, drafts_rounds = 0;
    // per draft position (the 1st .. 8th of a round): verified with all the drafts before it accepted, and accepted
    int64_t draft_reached[8] = {}, draft_kept[8] = {};
    double draft_ms = 0.0, verify_ms = 0.0;   // the drafter's time, the verification windows' time
    // DS4_MULTI_ATTN (with DS4_BATCH_DECODE): batch windows' member-layer attention passes taken by the shared launches, and
    // by the per-run path (the flag 0, or a window the shared launch does not take)
    int64_t mattn_shared = 0, mattn_per_run = 0;
};

class Model {
public:
    Model();
    ~Model();
    bool load(const Options& o, std::string& err);

    /// Feeds `token` at position pos(); with logits, the next token's logits are left on the head's device.
    bool step(int token, bool logits, std::string& err);
    /// Feeds n tokens (windows of up to 16, or a long run in chunks of prefill_chunk); the last `logits_rows` of them get
    /// logits (logits_rows > 16: only a run of one chunk or window).
    bool feed(const int* tokens, int n, int logits_rows, std::string& err);
    /// feed() of a prompt's end that also keeps the state before its last token in checkpoint `slot` (a prompt's
    /// last token then rides along in the run of chunks instead of a window of its own); the same as
    /// feed(n - 1), checkpoint(slot), feed(1)
    bool feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err);
    /// The next token from the last feed's last logit row (greedy when temperature <= 0).
    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);
    /// The last feed's logit rows on the host (rows x vocab floats).
    bool logits(std::vector<float>& out);

    /// The DSpark drafter (speculative decoding): its block size (0: not loaded), drafts for the positions after `y`
    /// (y: the token at pos(), not fed yet) with their confidences, and the verification of [y, drafts]: how many of the
    /// drafts stand and the token after them (the state then holds y and the accepted drafts).  The drafts are the
    /// drafter's argmax; a sampled request passes its chain (the same temperature / top_p / top_k / min_p / seed /
    /// counter as its verify()) so that with DS4_DSPARK_SAMPLE=1 they are drawn from the drafter's distribution, as
    /// DeepSeek's reference does.  verify() keeps a draft d with probability min(1, p(d) / q(d)) over the request's
    /// chain - the speculative-sampling rule, so every token is still distributed as the model's own draw.
    int draft_block() const;
    /// every GPU the engine uses is a gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) in a HIP build, false anywhere else (decided once)
    bool gpus_are_gfx12() const;
    bool draft(int y, std::vector<int>& drafts, std::vector<float>& conf, std::string& err, float temperature = 0.0f,
               float top_p = 1.0f, int top_k = 0, float min_p = 0.0f, uint64_t seed = 0, uint64_t counter = 0);
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err);

    /// Tokens a feed call takes at a time to use the chunked prompt path (all of a prompt: its chunks run as a
    /// pipeline; 16 without it).
    int feed_unit() const;
    /// Called from feed() while a run of chunks goes through the layers, with the tokens done so far.
    void set_progress(std::function<void(int64_t)> f);
    /// DS4_PF_HELPER_MAXP: the new tokens of the prompt read the next feeds make up (the server's read, which its pin may
    /// split in two feeds), so the read takes one decision; 0 ends the declaration (each feed is then judged by its own n).
    /// No effect unless DS4_PF_HELPER and DS4_PF_HELPER_MAXP are both live.
    void set_prompt_total(int64_t n);
    /// Asked while a run of chunks goes through the layers: true (a Stop) ends it early - one GPU after the layer it is
    /// on, several at a chunk all of them reach - and feed() returns false with cancelled() true.  The state past the
    /// prompt's start is then not usable: restore() a checkpoint kept before the feed, or reset().
    void set_cancel(std::function<bool()> f);
    bool cancelled() const;   // the last feed() / feed_mark() ended early on a Stop
    int64_t pos() const;
    int64_t max_context() const;
    int vocab() const;
    int dim() const;
    const std::vector<int32_t>& tokens() const;   // what the state has consumed, in order

    void reset();
    /// The post-prompt cache settle run on a thread of its own while the server waits for its next request (WP-8's option 1):
    /// the swap rounds of cache_settle without the store preload, stopped early when `cancel` goes true.  `done` is set when
    /// the thread has nothing left to do.  Only one such run at a time; the caller joins the thread after setting cancel.
    void settle_idle(const std::atomic<bool>& cancel);
    /// Keep the state at the current position in checkpoint slot 0 or 1 (the end of a prompt, the position before it);
    /// restore(limit) goes back to the latest one at or before limit (-1: none).
    void checkpoint(int slot = 0);
    bool restore(int64_t limit);
    int64_t restorable(int64_t limit) const;
    /// Checkpoint 1 also becomes the anchor (kept until a restore goes before it): the position before the end of the
    /// first prompt of a run that continued in place, where a re-rendered history (earlier reasoning dropped) splits.
    void pin();
    /// The positions restore() can go back to, latest first (a slot keeps them: it can be taken back at any of them).
    std::vector<int64_t> check_positions() const;

    /// The conversation in the model into `out` (slot_save): its position, its tokens, every window ring (the DSpark
    /// blocks' included), the compressors' partial groups and each KV source's compressed KV and index keys up to it,
    /// the pieces the state a prompt continues is made of.  0 and err: nothing was set aside (it holds pictures, there
    /// is no conversation, or the conversation is already set aside - unchanged since).
    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err);
    /// `data` (a slot's bytes) back into the model: the state at the slot's position n_pos, as slot_save took it -
    /// what a prompt that continues the slot's tokens reads on from (the caller checked the tokens are a prefix).
    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err);
    /// What a saved conversation must match: this engine's version, this model's geometry and files and the drafter's
    /// blocks - a slot naming anything else is not this model's state (a file's .meta says so; slot_load refuses).
    std::string slot_identity() const;

    /// DS4_BATCH_DECODE=N (docs/dsv41/cards/SV-03.md; unset or 0: off, nothing allocated): N sequence banks beside the
    /// conversation in the model, each a whole state, decoded together - a batch step is one window of B rows, row b
    /// bank b's next token, the dense weights and the experts read once for all of them.  Greedy only, no drafter, no
    /// pictures; one tensor-parallel group holding every layer (tp3 on 3 cards), else it says so and stays off.
    int batch_slots() const;                      // the banks (0: off)
    int64_t batch_pos(int bank) const;            // a bank's position (-1: no such bank)
    /// A prompt into bank `bank` from token 0 (the normal prompt path, on the bank's state); first: its greedy next token.
    /// The conversation in the model is left as it was.
    bool batch_prefill(int bank, const int* tokens, int n, int& first, std::string& err);
    /// One batch step: row b feeds tokens[b] to bank banks[b]; next[b]: row b's greedy next token.  A bank's rows (several:
    /// a teacher-forced run of its tokens) are contiguous and take its next positions.
    bool batch_step(const int* banks, const int* tokens, int B, int* next, std::string& err);
    /// DS4_BATCH_DSPARK=1 (default off; needs the drafter loaded): drafts inside batch steps.  batch_draft: bank's
    /// greedy drafts after y (its own drafter state).  batch_verify: one window over S sequences, sequence s = bank banks[s]
    /// with nrows[s] rows (rows: [y, drafts] of each, in turn; kWin rows at most); accepted[s] drafts stand, next[s] is
    /// the token after them; each bank then holds y and its accepted drafts, and its drafter is seeded with them.
    bool batch_spec_ok() const;
    /// DS4_BATCH_DSPARK_MAX (default 2): the scheduler's rule - draft (batch_draft + batch_verify) in a step while at
    /// most this many sequences are active, else a plain batch_step (no drafts, no taps, no drafter seeding); decided step
    /// by step as sequences join and leave.  0 when batch drafts are off.  A drafter that sat out plain steps drafts from
    /// a ring missing those positions (fewer accepted drafts for up to a window's tokens; the output is exact either way).
    int batch_spec_max() const;
    bool batch_draft(int bank, int y, std::vector<int>& drafts, std::vector<float>& conf, std::string& err);
    bool batch_verify(const int* banks, const int* nrows, const int* rows, int S, int* accepted, int* next,
                      std::string& err);
    /// DS4_BATCH_DSPARK_SAMPLED (srvbatch12; the serve loop's flag, default off): batch_verify with each sequence's
    /// sampling.  A sequence with temperature <= 0 is batch_verify's, byte for byte (its drafts stand while the trunk's
    /// argmax agrees).  A sequence with temperature > 0 keeps its drafts by the speculative-sampling rule over its own
    /// chain, as the solo verify() does with the drafter's argmax drafts: q a point at each draft, so draft d stands with
    /// probability p(d) (p through top_k / top_p / min_p / temperature), a rejected row is drawn from the residual
    /// max(0, p - q), and with every draft kept the next row is drawn as sample() would, at counter + drafts.  Row i of
    /// a sequence draws at counter + i: each kept token is distributed as the trunk's own draw.
    struct BatchSampling {
        float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
        int top_k = 0;
        uint64_t seed = 0, counter = 0;
    };
    bool batch_verify_sampled(const int* banks, const int* nrows, const int* rows, int S, const BatchSampling* samp,
                              int* accepted, int* next, std::string& err);
    /// srvbatch (the serve loop's continuous batching): bank `bank`'s whole state and the conversation in the model
    /// trade places - bank_swap's pointer swap after a queue drain, no copy, its own inverse.  The prompt path, the
    /// checkpoints, restore() and slot_save / slot_load then work on what was the bank, so a batch admission reads its
    /// prompt with every reuse the solo path has and is then swapped into its bank.  Between windows only, and never for
    /// a bank that is decoding (the caller's rule).
    bool bank_swap(int bank, std::string& err);
    /// What bank `bank` has consumed (null: no such bank) - the serve loop's prefix match.
    const std::vector<int32_t>* bank_tokens(int bank) const;
    /// Row `row` of the last batch_step's window sampled with these parameters (the solo sample()'s kernel, on that row):
    /// a request with temperature > 0 in a plain batch step.  Greedy rows keep batch_step's pick (the same argmax).
    int batch_sample(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);

    /// The vision tower: the token every position of an image's span carries (-1: no tower).
    int image_token() const;
    /// An image's pixels (RGB rows, width x height, both multiples of the patch size: the server's resize) -> the
    /// rows of its span, BF16 [n_h * (n_w + 1) + 2][dim]: image_start, n_h rows of (n_w aligner rows, image_newline),
    /// image_end.
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err);
    /// The pictures of the prompt about to be fed ({}: none): positions inside a span read its rows (and route with
    /// the image bias, and take no part in the Engram n-grams); the rest the embedding table's.
    void set_images(std::vector<Image> images);

    void save_profile() const;
    const Stats& stats() const;
    std::string describe() const;
    /// The helpers' lines (fill, then the run's counters: experts, windows, host ms issuing + waiting); "" without one.
    std::string helper_report() const;
    /// INFO / VRAM lines for the server's Monitor.
    std::vector<std::string> info_lines() const;

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4
