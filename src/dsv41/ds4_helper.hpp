// Strata prototype (docs/dsv41/cards/EB-01.md, phase 1). src/dsv41/ds4_helper.hpp - a helper expert cache for the
// DeepSeek-V4.1 engine: one extra GPU (an MI50, gfx906, wave64 - or any device the binary has code for) that owns no
// layers and holds a profile-ranked set of the routed experts the primary's VRAM does not.  It is a second answerer on
// the CPU lane's mailbox (ds4_kernels.cuh Req / Resp): for a request of layer li it takes the picks it owns, copies the
// window's FFN inputs and a plan to its own memory (no zero-copy: mapped host reads measured slow on the MI50 in the
// main engine), runs the same gg::win_experts the primary's window path runs (q8 of x, gate/up, q8 of h, down, the
// weighted per-token combine), copies the T x n_embd result back through pinned memory and adds it into Resp::part[t]
// before the lane publishes Resp::seq.  The primary's device code is untouched: its combine adds cpu_part[t] when
// cpu[t] is set, whoever filled it.
//
// Phase 1 restrictions (checked at open / fill, printed): GGUF packs only (IST-DAS i-quants or UD MXFP4 experts; no
// EXL3), trunk layers only (no drafter blocks), decode / verify windows
// through the lane only (the sync host-routed window path and prefill leave helper-owned experts to the CPU lane,
// which still reads them from the files: the helper holds a copy, nothing moves).  Under tensor parallel (--tp > 1) the
// helper still takes WHOLE experts (read from the GGUF through member 0's whole-blob layout) and answers the lead's request
// only; its row is added to the lead's CPU-lane part, which the lead's combine sums once before the all-reduce.  Several helpers
// (HelperSet) change nothing there: each pick goes to the one helper that owns it, the lane thread adds every helper's row into
// the same Resp::part before it publishes Resp::seq, and the non-lead members pass no cpu_part.  Off by default; with it off the
// engine's behaviour is unchanged (every hook is behind enabled()).
#pragma once

#include "ds4_gguf.cuh"     // x3::ExpertLayout, gg::
#include "ds4_kernels.cuh"  // Req, Resp, kWin, kPicks, kMaxDim

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4::helper {

namespace gg = strata::ds4::gguf;   // (the aliases ds4_model.cu keeps in strata::ds4 are not visible here)
namespace x3 = gg::x3;

struct Config {
    int device = -1;          // the helper's CUDA / HIP ordinal; -1: off
    int64_t slots = 0;        // experts to hold; 0: as many as fit (free memory less reserve_mib and the scratch)
    int64_t reserve_mib = 512;
    bool verbose = false;
};

/// Device acceptance.  "" when `ordinal` may take the role, else the reason.  In a HIP build a wave64 device is
/// accepted only as a helper, and only when this binary carries its architecture (STRATA_HIP_HELPER_ARCHS, the fat
/// binary of cmake/hip_backend.cmake); as a primary (a --gpus member) it is refused.  Other backends: always "".
std::string accept_device(int ordinal, bool as_helper);

/// What the helper needs to know of a layer (read from the primary's Layer at fill time, never written).
struct LayerGeo {
    int li = 0;             // index in the model's layer list (what the lane's layer_of carries)
    int n_exp = 0, k_exp = 0;
    size_t blob = 0;        // bytes of one expert's blob: gate | up | down at lay.trellis[r], the VRAM slot layout
    x3::ExpertLayout lay;   // (fmt gg::kFmtMXFP4 with lay.K[r] the roles' ggml types)
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    bool eligible = true;   // false: the helper never holds this layer's experts (mtp, cpu_only, not GGUF)
    int prio = 0;           // candidates of a higher prio rank before any lower (DS4_DRAFT_EXPERTS=helper: the drafter's 1)
};

/// One of a request's picks the helper owns: token t's pick i (Req::ids[t][i], weight Req::w[t][i]) is expert e.
struct Pick {
    int t = 0, i = 0, e = 0;
};

struct Stats {
    int64_t slots = 0;             // experts held
    size_t bytes = 0;              // their memory
    int64_t experts = 0;           // entries computed (a token x expert)
    int64_t windows = 0;           // requests with at least one helper pick
    int64_t tokens = 0;            // tokens whose part the helper added to
    double begin_ms = 0.0;         // host time in begin() (copies issued, launches)
    double wait_ms = 0.0;          // host time in finish() (the event wait and the adds)
};

class Helper {
public:
    Helper();
    ~Helper();
    Helper(const Helper&) = delete;
    Helper& operator=(const Helper&) = delete;

    bool enabled() const;
    int device() const;

    /// The device, its stream and event, the scratch and the pinned return buffer.  The current device is restored.
    bool open(const Config& c, std::string& err);

    /// Fill (load time, after the primary's VRAM slots are fixed): every (li, e) of an eligible layer that
    /// on_primary(li, e) is false for is a candidate, ranked by its profile share (prof[li * prof_stride + e] over its
    /// layer's sum; a layer without counts ranks by index); the first `slots` (Config) go in, each read by
    /// read(li, e, dst) into a pinned staging blob of layers[li].blob bytes and copied to the helper.  Blobs are
    /// slots sized per layer (each expert takes its layer's blob, 256-byte aligned; DS4_HELPER_UNIFORM_SLOTS=1 restores
    /// uniform slots of the largest layer blob).  Ownership is then owns(li, e).
    using OnPrimary = std::function<bool(int li, int e)>;
    using Reader = std::function<bool(int li, int e, uint8_t* dst)>;
    bool fill(const std::vector<LayerGeo>& layers, const uint32_t* prof, size_t prof_stride, const OnPrimary& on_primary,
              const Reader& read, std::string& err);

    /// The two halves of fill(), for HelperSet (which ranks once and deals the ranks out over several helpers):
    /// prepare() takes the layer table and the scratch and fixes capacity() (Config::slots, or what fits); load()
    /// takes the given (li, e) pairs, slot i the i-th, and reads them in.  fill() is prepare + the first
    /// min(capacity, candidates) ranked + load.
    bool prepare(const std::vector<LayerGeo>& layers, std::string& err);
    int64_t capacity() const;   // (an upper bound on the count: Config::slots, or room / the smallest slot)
    /// Whether one more expert of layer li fits when `used_bytes` / `used_slots` are already taken (Config::slots caps the
    /// count, else the free memory less the reserve caps the bytes); on true both are advanced by this expert.  The fill
    /// walks the ranked candidates with it (rank order: the first that does not fit ends the helper's list).
    bool take(int li, size_t& used_bytes, int64_t& used_slots) const;
    bool load(const std::vector<std::pair<int, int>>& assigned, const Reader& read, std::string& err);

    /// A second lane on this helper's experts (own stream, event, scratch; the arena shared read-only), for the pool
    /// server thread.  null with `err` on failure.  Destroy before this helper.
    std::unique_ptr<Helper> make_view(std::string& err) const;
    /// Synchronous compute for the pool server: the picks (all owned, layer li) of T tokens, x rows kMaxDim apart in any
    /// host memory, w[t][i] their weights; out[t * kMaxDim ..) gets token t's weighted sum (n_embd floats) and has[t]
    /// = 1 for the tokens with a pick.  Meant for a view (the lane thread's begin / finish own the first lane).
    bool serve(int li, const float* x, const float (*w)[kPicks], int T, const Pick* picks, int n, float* out, uint8_t* has);
    const std::vector<LayerGeo>& layer_table() const;

    /// One request of a pool pass (a layer's picks of T tokens, x rows kMaxDim apart in any host memory).  After
    /// pass_finish(): ok and out[t * kMaxDim ..) / has[t] as serve() leaves them.
    struct PoolReq {
        int li = 0;
        const float* x = nullptr;
        const float (*w)[kPicks] = nullptr;
        int T = 0;
        const Pick* picks = nullptr;
        int n = 0;
        float* out = nullptr;
        uint8_t* has = nullptr;
        bool ok = false;
    };
    /// Asynchronous pool pass on a view: the requests (sum of T <= kWin, in arrival order) as one set of launches per layer
    /// type (union batching: requests whose layers have the same geometry and expert layout share a window, an expert two
    /// of them pick is read once), one H2D of the plans, one of the x rows, one D2H, one event.  false: nothing issued
    /// (every request's ok stays false).  A request without a held pick is not issued (ok false).
    bool pass_begin(PoolReq* rq, int n);
    bool pass_busy() const;
    /// 0: still running, 1: done (pass_finish() copies out), -1: a GPU error (every request's ok is cleared).
    int pass_poll();
    void pass_finish();

    bool owns(int li, int e) const;
    /// Experts held of layer li.
    int held(int li) const;

    /// Dispatch, on the lane thread: the request's helper picks for layer li (n of them, every one owned); copies and
    /// launches are issued on the helper's stream, the event recorded; nothing waits.  The caller's current device is
    /// restored.  false: nothing issued (n == 0, not enabled, or a failure that was printed - the picks are then NOT
    /// computed by anyone; the caller should treat them as CPU-lane picks, see fallback()).
    bool begin(int li, const Req& r, int T, const Pick* picks, int n);
    /// DS4_HELPER_BATCH=1 (docs/dsv41/cards/HX-10.md): share_x() points this helper's fetch at HelperSet's shared staging of
    /// the request's x rows (before any window; false: it keeps its own copy); begin_staged() is begin() with the rows already
    /// written there (the lane copies them once a request, not once a helper).
    bool share_x(uint8_t* host);
    bool x_shared() const;
    bool begin_staged(int li, const Req& r, int T, const Pick* picks, int n);
    /// Join, on the lane thread, after the CPU lane's part is in rp: waits for the event, then for each token with a
    /// helper pick adds the helper's row into rp.part[t] (copies it when rp.cpu[t] == 0) and sets rp.cpu[t] = 1.
    void finish(Resp& rp, int T);
    /// Whether the last begin() failed to issue (its picks need the CPU lane).
    bool fallback() const;
    /// DS4_HELPER_ARMED=1 (HIP, kfetch + zc + flag; opt-in, off by default; docs/dsv41/cards/HX-09.md): arm() launches the window of
    /// request `seq` (layer li, T rows) before the request exists - its head waits on the device for the lane's command -
    /// so the begin() that serves that request only stages the plan and x and writes the command (no launch). A begin()
    /// for any other request first cancels the armed window (it runs nothing), then issues as before. arm_drop() cancels
    /// it (the lane calls it after each request's begin, for helpers without picks). A head that hears nothing within
    /// DS4_HELPER_ARMED_MS (50) gives up; finish() then re-issues the window. Windows of more than DS4_HELPER_ARMED_MAXROWS
    /// (4) rows are not armed (they take the path as before). The lane's thread only.
    bool armed_on() const;
    bool arm(int li, int T, unsigned seq);
    void arm_drop();

    /// DS4_HELPER_DD=1 (docs/dsv41/cards/HX-17.md; HIP, kfetch + zc + flag + graph; default off): a
    /// persistent device-direct window.  dd_arm() launches layer li's window for request `seq` before it exists: its head
    /// waits on the device for the request ring's slot (reqs, fine-grained host memory) to carry `seq`, then reads the picks,
    /// weights and x rows itself, builds the plan exactly as begin() does (only the experts take(li, e) left it: dd_open's
    /// table) and writes has[t]; the body (the (layer, T) graph, no fetch) writes the rows into the zc out buffer; a flag
    /// kernel then writes seq into the view's seq word.  The primary's lead waits for every helper's seq on the device and
    /// joins the rows itself (the lane never touches them).  The lane's thread only.
    struct DdView {
        const volatile unsigned long long* seq = nullptr;   // host (fine-grained): the last request answered
        const int* has = nullptr;                           // host: [kWin] token t has a row
        const float* out = nullptr;                         // host: [kWin][n_embd] the rows
    };
    using Take = std::function<bool(int li, int e)>;
    bool dd_open(const Req* reqs, const Take& take, std::string& err);
    bool dd_on() const;
    DdView dd_view() const;
    bool dd_arm(int li, int T, unsigned seq);
    /// Whether the window armed last is the one for (seq, li, T).
    bool dd_armed(unsigned seq, int li, int T) const;
    /// No window armed for seq: the lane answers for this helper (has = 0, then seq) - its picks go to the CPU lane.
    void dd_answer_host(unsigned seq, int T);
    /// DS4_HELPER_DD_SKIP: whether expert e of layer li is in this helper's device-direct table (its head takes the pick)
    bool dd_owns(int li, int e) const;

    /// DS4_PF_HELPER (see pf_helper_rows): a prompt chunk's picks of this helper's experts, computed on its own stream.
    /// pf_open(): the buffers for chunks of up to max_tok rows and the per-layer tables; x_host: HelperSet's fine-grained
    /// staging of the chunk's FFN inputs (F32 [max_tok][n_embd]).  pf_local(li, e): the expert's index in layer li's strided
    /// run, -1 not held (or the layer not laid out as one run).  pf_issue(): the picks (t, i, e) of T rows, w[t * kPicks + i]
    /// the routing weights (host), issued - the rows of x the picks need fetched, gate/up, down (the prompt sets' light
    /// kernels, in rounds of <= 16 rows an expert) and each token's weighted sum in pick order into the fine-grained out rows
    /// (out[j] for the j-th token with a pick; map[t] = j, -1 none).  pf_wait(): its event.
    bool pf_open(int max_tok, const float* x_host, std::string& err);
    bool pf_ready() const;
    int pf_local(int li, int e) const;
    bool pf_issue(int li, int T, const Pick* picks, int n, const float* w, std::string& err);
    int pf_tokens() const;   // the last issue's tokens (0: nothing to wait for or add)
    bool pf_wait();
    const float* pf_out() const;   // fine-grained host [max_tok][n_embd]
    const int* pf_map() const;     // fine-grained host [max_tok]
    std::string pf_describe() const;

    const Stats& stats() const;
    /// The run's counters back to zero (slots and bytes kept): after the load's warm-up rounds.
    void reset_stats();
    /// One line for describe(): device, slots, GiB, layers covered.
    std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// STRATA_POOL_CLIENT (the directory list, "dir[,dir...]") or null.
const char* pool_client_dirs();

/// DS4_PF_HELPER=<rows> (docs/dsv41/cards/PF-11.md; default unset = off): the rows a prompt chunk may have for its
/// helper-held experts to be computed on the helpers instead of streamed to the primaries.  0: off.  Set at all, it also
/// makes each helper lay out its experts layer by layer, expert index order (Helper::load), so a layer's experts are one
/// strided run the prompt sets' kernels can index - the decode path reads them through pointers, so its bits do not move.
int pf_helper_rows();

/// Several helpers (--helper-device 1,4).  The experts no primary slot holds are ranked once (profile share, as for one
/// helper); the helpers between them hold the ranked prefix that fits.  How it is placed (DS4_HELPER_PLACE):
///  - layer (the default with 2+ helpers): every layer's held experts on ONE helper, so a request (one layer) goes to one
///    helper and the lane pays one issue + wait a layer, not one a helper.  Layers are dealt heaviest first (a layer's
///    load: the expected picks a token of its held experts, share x k_exp from the same profile) to the helper whose
///    load / weight stays lowest among those the layer fits; weights DS4_HELPER_WEIGHTS=a,b,... (one per helper, e.g.
///    1,0.7 for a slower second card), by default each helper's slot capacity.  Each helper then takes its layers'
///    candidates in rank order until full; what does not fit stays on the CPU lane.  A "helper: placement" line gives
///    each helper's layers and expected picks a token.
///  - stripe: as the main engine's --expert-cache-device1..3 (RemoteExperts, --expert-cache-remote-placement stripe):
///    rank 0 to the first helper, rank 1 to the second, ... and back to the first; a helper that is full (its slots, or
///    its memory) is skipped, so the rest go to the others.
/// Both are deterministic (the same profile and devices give the same placement; the parts are added in helper order).
/// Each helper has its own device, stream, event and scratch; begin() issues to every helper that owns picks of the
/// request (all in flight at once), finish() joins them in order.  With one helper it is that helper, as before.
///
/// The shared pool (docs/dsv41/cards/HX-03.md), same environment as the main engine: with STRATA_POOL_SERVE=dir[,dir] the n-th
/// helper also answers other dsv41 engines through the /dev/shm mailbox directory (its own stream and thread, plan.bin
/// published once it is filled); with STRATA_POOL_CLIENT=dir[,dir] the n-th entry of the device list is NOT opened:
/// it is a client of the helper another engine serves in that directory (device ordinals are then only counted).
class HelperSet {
public:
    HelperSet();
    ~HelperSet();
    HelperSet(const HelperSet&) = delete;
    HelperSet& operator=(const HelperSet&) = delete;

    /// base.device is ignored; one helper per entry of `devices` (each one a distinct ordinal).
    bool open(const std::vector<int>& devices, const Config& base, std::string& err);
    bool enabled() const;
    size_t size() const;
    bool fill(const std::vector<LayerGeo>& layers, const uint32_t* prof, size_t prof_stride, const Helper::OnPrimary& on_primary,
              const Helper::Reader& read, std::string& err);
    bool owns(int li, int e) const;
    /// Issue the picks (every one owned by some helper) to their helpers.  Returns true when at least one helper took
    /// its picks; the picks no helper took (an issue failure, printed) are added to `rejected` - the caller hands
    /// those to the CPU lane.
    bool begin(int li, const Req& r, int T, const Pick* picks, int n, std::vector<Pick>& rejected);
    /// Join every helper that took picks (adds their parts into rp as Helper::finish does).  A pool client whose server
    /// did not answer within 5 s (or refused) is dropped for good (owns() false from then on) and its picks of this
    /// request are added to `failed`: the caller computes them on the CPU lane and adds the result into rp.
    void finish(Resp& rp, int T, std::vector<Pick>& failed);
    /// DS4_HELPER_DEFER two windows in flight (docs/dsv41/cards/HX-12.md): open_deferred() gives every local helper a second
    /// lane on its own experts (Helper::make_view: own stream, staging and scratch, the arena shared read-only), after fill.
    /// begin_deferred() / finish_deferred() are begin() / finish() on those lanes, so a layer's deferred picks run on the
    /// MI50 beside its kept ones instead of behind them.  Not for a pool client; false / no lanes: the caller keeps the
    /// one-lane path.  (Armed windows stay on the first lanes only.)
    bool open_deferred(std::string& err);
    bool deferred_ready() const { return !hd_.empty() && hd_.size() == hs_.size(); }
    bool begin_deferred(int li, const Req& r, int T, const Pick* picks, int n, std::vector<Pick>& rejected);
    void finish_deferred(Resp& rp, int T);
    /// DS4_HELPER_ARMED: Helper's, over every local helper (none for a pool client). arm(): true if any helper armed.
    bool armed_on() const;
    bool arm(int li, int T, unsigned seq);
    void arm_drop();
    /// DS4_HELPER_DD (Helper's, over every local helper; not for a pool client).  dd_open(): each pick goes to the first
    /// helper that owns it (as begin() deals them) and only experts on_primary() is false for; true: every helper took it.
    bool dd_open(const Req* reqs, const Helper::OnPrimary& on_primary, std::string& err);
    bool dd_on() const { return dd_on_; }
    std::vector<Helper::DdView> dd_views() const;
    /// Arm every helper for (li, T, seq); false if any failed (printed; dd_check() then answers for it).
    bool dd_arm(int li, int T, unsigned seq);
    /// On the request seq (layer li, T rows) the lead joins on the device: every helper not armed for it is answered by the
    /// host (has = 0) and its picks among `picks` are moved to `cpu` (the CPU lane computes them).  Returns the helpers answered so.
    int dd_check(unsigned seq, int li, int T, const std::vector<Pick>& picks, std::vector<Pick>& cpu);
    int64_t dd_armed_n = 0, dd_host_n = 0;   // (the run's counters)
    /// DS4_HELPER_DD_SKIP: [li * n_exp + e] the index (in dd_views() order) of the helper whose head takes expert e of layer
    /// li, -1 none - the heads' own tables, so a helper without an entry for any of a request's picks writes has = 0 for every row
    std::vector<int8_t> dd_owner_table(int n_layers, int n_exp) const;
    /// DS4_PF_HELPER (see pf_helper_rows; HIP, local helpers only): pf_open() gives every helper its buffers and the shared
    /// fine-grained x staging (false: off, err says why).  pf_owner(li, e): the helper (index) that takes expert e of layer li in
    /// a prompt chunk - the first that holds it in its strided run, as begin() deals a pick to the first owner - or -1.
    /// pf_issue() (one thread, the group lead's): the T rows' picks (ids / weights host [T][kPicks], k a row) of the experts
    /// host(e) is true for, each to its owner, issued; an owner-less one is an error.  pf_tokens / pf_wait / pf_out / pf_map: Helper's.
    bool pf_open(int max_tok, std::string& err);
    bool pf_on() const { return pf_on_; }
    float* pf_x() const { return pf_x_; }
    int pf_owner(int li, int e) const;
    bool pf_issue(int li, int T, int k, const int* ids, const float* w, const std::function<bool(int)>& host, std::string& err);
    int pf_tokens(size_t h) const { return h < hs_.size() ? hs_[h]->pf_tokens() : 0; }
    bool pf_wait(size_t h) { return h < hs_.size() && hs_[h]->pf_wait(); }
    const float* pf_out(size_t h) const { return h < hs_.size() ? hs_[h]->pf_out() : nullptr; }
    const int* pf_map(size_t h) const { return h < hs_.size() ? hs_[h]->pf_map() : nullptr; }
    std::string pf_describe() const;
    /// One line a helper (the fill line, plus the run's counters once it has served).
    std::string describe() const;
    /// Every helper's (and pool client's) run counters to zero (after the load's warm-up).
    void reset_stats();
    Helper& at(size_t i);
    const Helper& at(size_t i) const;
    bool pool_client() const { return client_; }
    /// build 8: the placement when DS4_HELPER_PLACE is unset - stripe if set (the engine sets it for a tensor-parallel
    /// group of 3+ members with 2+ helpers: the test machine tp3 + 3 MI50, drafted solo +7-9 %), else layer
    bool stripe_default = false;

private:
    struct Remote;
    struct Server;
    bool fill_client(const std::vector<LayerGeo>& layers, const Helper::OnPrimary& on_primary, std::string& err);
    bool start_servers(const std::vector<LayerGeo>& layers, std::string& err);
    void stop_servers();
    std::vector<std::unique_ptr<Helper>> hs_;
    std::vector<uint8_t> issued_;   // by helper: took picks in the request in flight
    std::vector<std::vector<Pick>> by_;   // begin()'s picks by helper / server, kept (the lane's thread; no allocation a request)
    uint8_t* xs_host_ = nullptr;          // DS4_HELPER_BATCH: the shared x staging (fine-grained, every helper maps it)
    std::vector<std::unique_ptr<Helper>> hd_;   // DS4_HELPER_DEFER: the deferred lanes (views of hs_[k]), destroyed before hs_
    std::vector<uint8_t> issued_d_;             // (by helper: its deferred lane took picks in the request in flight)
    bool client_ = false;
    bool dd_on_ = false;   // DS4_HELPER_DD
    bool pf_on_ = false;   // DS4_PF_HELPER
    float* pf_x_ = nullptr;                      // (fine-grained host, every helper maps it)
    std::vector<std::vector<Pick>> pf_by_;       // (pf_issue's picks by helper, kept)
    std::vector<std::unique_ptr<Remote>> rem_;
    std::vector<std::unique_ptr<Server>> srv_;   // (after hs_: destroyed first, its views before their sources)
};

}  // namespace strata::ds4::helper
