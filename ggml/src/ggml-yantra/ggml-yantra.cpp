// ggml-yantra: a ggml backend whose device is yantra (nixpt/yantra#211).
//
// Device: yantra through its C ABI (yantra-capi sessions). YANTRA_DEVICE
// picks it: `local` is an in-process yantra node on the CPU driver,
// `remote:SPEC` a rented node. Unset, the backend registers no device and
// nothing changes.
//
// Buffers are host memory ("host-shadowed"): the CPU backend runs every op
// yantra does not, on the same bytes, with no copies. A weight a supported
// op uses is uploaded to yantra once, content-addressed and resident, and
// named by id afterwards; only activations go up and only results come down.
//
// One exchange per graph split: activation uploads and jobs are queued in
// the session, and one download_many of the split's outputs sends them all.

#include "ggml-yantra.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "yantra.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#define YANTRA_BUFFER_ALIGNMENT 64

// Quantized MUL_MAT runs as one yantra QGemv per activation row; past this
// many rows (prefill) the op stays on the CPU. GGML_YANTRA_QGEMV_ROWS
// overrides it.
#define YANTRA_QGEMV_MAX_ROWS 8

// A QGemv job is charged 2 credits per output row against the node's
// budget (4096 per vCPU), so a large weight (an LM head of 150k rows) is
// split into row chunks, each its own resident buffer and job. 8192 rows
// fit any node of 4+ vCPUs. GGML_YANTRA_ROW_CHUNK overrides it.
#define YANTRA_ROW_CHUNK 8192

// ---------------------------------------------------------------------------
// The process-wide session and the resident-weight cache.
// ---------------------------------------------------------------------------

static bool yantra_enabled() {
    const char * v = std::getenv("YANTRA_DEVICE");
    return v != nullptr && *v != '\0';
}

static std::mutex g_mutex;
static YantraSession * g_session = nullptr;
static bool g_session_tried = false;
// Host address of a weight tensor's data -> its resident yantra buffer.
// A quantized weight is held as row chunks (see yantra_row_chunk), one id each.
static std::unordered_map<const void *, std::vector<uint64_t>> g_resident;

// Opened on first use, closed at process exit (yantra frees everything).
static YantraSession * yantra_session() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session_tried) {
        g_session_tried = true;
        g_session = yantra_session_open();
        if (g_session == nullptr) {
            GGML_LOG_ERROR("%s: cannot open a yantra session (YANTRA_DEVICE=%s)\n", __func__,
                           std::getenv("YANTRA_DEVICE"));
        }
    }
    return g_session;
}

// Forget cached weights that lie in [base, base + size): their bytes changed
// or their buffer is going away. (A freed resident buffer stays on the
// device until the session closes; the capi does not free resident ones.)
static void yantra_forget(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const char * lo = (const char *) base;
    for (auto it = g_resident.begin(); it != g_resident.end();) {
        const char * p = (const char *) it->first;
        if (p >= lo && p < lo + size) {
            it = g_resident.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer: host memory.
// ---------------------------------------------------------------------------

struct ggml_backend_yantra_buffer_context {
    void * data;
    size_t size;
};

static void ggml_backend_yantra_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_yantra_buffer_context *) buffer->context;
    yantra_forget(ctx->data, ctx->size);
    ggml_aligned_free(ctx->data, ctx->size);
    delete ctx;
}

static void * ggml_backend_yantra_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_yantra_buffer_context *) buffer->context)->data;
}

static void ggml_backend_yantra_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                     size_t offset, size_t size) {
    yantra_forget(tensor->data, ggml_nbytes(tensor));
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_yantra_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                  size_t offset, size_t size) {
    yantra_forget(tensor->data, ggml_nbytes(tensor));
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_yantra_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                  size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_yantra_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        yantra_forget(dst->data, ggml_nbytes(dst));
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void ggml_backend_yantra_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_yantra_buffer_context *) buffer->context;
    yantra_forget(ctx->data, ctx->size);
    memset(ctx->data, value, ctx->size);
}

static const ggml_backend_buffer_i ggml_backend_yantra_buffer_i = {
    /* .free_buffer     = */ ggml_backend_yantra_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_yantra_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_yantra_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_yantra_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_yantra_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_yantra_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_yantra_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// ---------------------------------------------------------------------------
// Buffer type.
// ---------------------------------------------------------------------------

static const char * ggml_backend_yantra_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "Yantra";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_yantra_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size = size == 0 ? 1 : size;
    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, size);
        return nullptr;
    }
    auto * ctx = new ggml_backend_yantra_buffer_context{ data, size };
    return ggml_backend_buffer_init(buft, ggml_backend_yantra_buffer_i, ctx, size);
}

static size_t ggml_backend_yantra_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return YANTRA_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool ggml_backend_yantra_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_yantra_buffer_type();

static bool ggml_backend_buft_is_yantra(ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_yantra_buffer_type();
}

// ---------------------------------------------------------------------------
// Backend: graph compute.
// ---------------------------------------------------------------------------

struct ggml_backend_yantra_context {
    // Device scratch buffers by size, reused across graphs so a steady-state
    // split allocates nothing (an alloc is a round trip of its own).
    std::unordered_map<uint64_t, std::vector<uint64_t>> pool;
    std::vector<std::pair<uint64_t, uint64_t>> in_use;  // (size, id)
    // This graph's pending reads, and the host ranges they will fill.
    std::vector<YantraRead> reads;
};

static bool yantra_take(ggml_backend_yantra_context * ctx, YantraSession * s, uint64_t size, uint64_t * id) {
    auto & free_list = ctx->pool[size];
    if (!free_list.empty()) {
        *id = free_list.back();
        free_list.pop_back();
    } else if (yantra_session_alloc(s, size, id) != YANTRA_OK) {
        return false;
    }
    ctx->in_use.emplace_back(size, *id);
    return true;
}

static void yantra_release_all(ggml_backend_yantra_context * ctx) {
    for (auto & [size, id] : ctx->in_use) {
        ctx->pool[size].push_back(id);
    }
    ctx->in_use.clear();
}

// Send the queue and fill every pending output.
static bool yantra_flush(ggml_backend_yantra_context * ctx, YantraSession * s) {
    int rc = yantra_session_download_many(s, ctx->reads.data(), (uint32_t) ctx->reads.size());
    ctx->reads.clear();
    if (rc != YANTRA_OK) {
        GGML_LOG_ERROR("%s: yantra exchange failed (rc=%d)\n", __func__, rc);
        return false;
    }
    return true;
}

// Whether `t`'s bytes are an output still waiting in `ctx->reads`.
static bool yantra_pending(const ggml_backend_yantra_context * ctx, const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    const char * lo = (const char *) t->data;
    const char * hi = lo + ggml_nbytes(t);
    for (const auto & r : ctx->reads) {
        const char * rlo = (const char *) r.dst;
        if (lo < rlo + r.len && rlo < hi) {
            return true;
        }
    }
    return false;
}

static uint64_t yantra_row_chunk() {
    static const uint64_t rows = [] {
        const char * v = std::getenv("GGML_YANTRA_ROW_CHUNK");
        const uint64_t r = v && *v ? (uint64_t) std::strtoull(v, nullptr, 10) : (uint64_t) YANTRA_ROW_CHUNK;
        return r > 0 ? r : (uint64_t) YANTRA_ROW_CHUNK;
    }();
    return rows;
}

// The device buffers holding `w`'s data in the layout the op wants:
// resident and cached when `w` is a weight, otherwise scratch uploads.
// - f32: one buffer, W^T (K x N): yantra's Matmul wants B = W^T, and ggml
//   stores W as N rows of K.
// - quantized: ggml's blocks are QGemv's layout, uploaded as they are, in
//   chunks of yantra_row_chunk() rows (the last one shorter).
static bool yantra_weight(ggml_backend_yantra_context * ctx, YantraSession * s, const ggml_tensor * w,
                          std::vector<uint64_t> & ids) {
    const bool is_weight = w->buffer && w->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    if (is_weight) {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_resident.find(w->data);
        if (it != g_resident.end()) {
            ids = it->second;
            return true;
        }
    }

    // (bytes, len) per chunk.
    std::vector<std::pair<const void *, size_t>> parts;
    std::vector<float> transposed;
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    if (w->type == GGML_TYPE_F32) {
        const float * src = (const float *) w->data;
        transposed.resize((size_t) (K * N));
        for (int64_t n = 0; n < N; n++) {
            for (int64_t k = 0; k < K; k++) {
                transposed[k * N + n] = src[n * K + k];
            }
        }
        parts.emplace_back(transposed.data(), transposed.size() * sizeof(float));
    } else {
        const size_t row_bytes = ggml_row_size(w->type, K);
        const uint64_t R = yantra_row_chunk();
        for (uint64_t r0 = 0; r0 < (uint64_t) N; r0 += R) {
            const uint64_t rows = std::min<uint64_t>(R, (uint64_t) N - r0);
            parts.emplace_back((const char *) w->data + r0 * row_bytes, rows * row_bytes);
        }
    }

    ids.clear();
    for (const auto & [bytes, len] : parts) {
        uint64_t id = 0;
        if (is_weight) {
            if (yantra_session_upload_resident(s, bytes, len, &id) != YANTRA_OK) {
                return false;
            }
        } else if (!yantra_take(ctx, s, len, &id) || yantra_session_upload(s, id, 0, bytes, len) != YANTRA_OK) {
            return false;
        }
        ids.push_back(id);
    }
    if (is_weight) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_resident[w->data] = ids;
    }
    return true;
}

static bool yantra_mul_mat(ggml_backend_yantra_context * ctx, YantraSession * s, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const uint64_t K = (uint64_t) src0->ne[0];
    const uint64_t N = (uint64_t) src0->ne[1];
    const uint64_t M = (uint64_t) src1->ne[1];

    std::vector<uint64_t> w;
    if (!yantra_weight(ctx, s, src0, w)) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                uint64_t x = 0, c = 0;
                if (!yantra_take(ctx, s, M * K * 4, &x) || !yantra_take(ctx, s, M * N * 4, &c)) {
                    return false;
                }
                if (yantra_session_upload(s, x, 0, src1->data, M * K * 4) != YANTRA_OK ||
                    yantra_session_matmul_f32(s, x, w[0], c, M, N, K) != YANTRA_OK) {
                    return false;
                }
                ctx->reads.push_back({ c, 0, M * N * 4, dst->data });
                return true;
            }
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            {
                // y_m = W x_m per activation row m and weight chunk: QGemv
                // names whole buffers, so each row gets its own (pooled) x,
                // and each (row, chunk) its own y, read into its place.
                const uint32_t fmt = src0->type == GGML_TYPE_Q8_0 ? YANTRA_Q8_0 : YANTRA_Q4_0;
                const uint64_t R = yantra_row_chunk();
                for (uint64_t m = 0; m < M; m++) {
                    uint64_t xm = 0;
                    const char * row = (const char *) src1->data + m * src1->nb[1];
                    if (!yantra_take(ctx, s, K * 4, &xm) || yantra_session_upload(s, xm, 0, row, K * 4) != YANTRA_OK) {
                        return false;
                    }
                    for (size_t c = 0; c < w.size(); c++) {
                        const uint64_t r0 = c * R;
                        const uint64_t rows = std::min<uint64_t>(R, N - r0);
                        uint64_t ym = 0;
                        if (!yantra_take(ctx, s, rows * 4, &ym) ||
                            yantra_session_qgemv(s, fmt, w[c], xm, ym, rows, K) != YANTRA_OK) {
                            return false;
                        }
                        ctx->reads.push_back(
                            { ym, 0, rows * 4, (char *) dst->data + m * dst->nb[1] + r0 * sizeof(float) });
                    }
                }
                return true;
            }
        default:
            return false;
    }
}

static enum ggml_status ggml_backend_yantra_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_yantra_context *) backend->context;
    YantraSession * s = yantra_session();
    if (s == nullptr) {
        return GGML_STATUS_FAILED;
    }
    bool ok = true;
    for (int i = 0; ok && i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node)) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_MUL_MAT:
                // An input that is an earlier output of this graph must be
                // read back before it is uploaded.
                if (yantra_pending(ctx, node->src[0]) || yantra_pending(ctx, node->src[1])) {
                    ok = yantra_flush(ctx, s);
                }
                ok = ok && yantra_mul_mat(ctx, s, node);
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                ok = false;
        }
    }
    // One exchange: every queued upload and job, and every output.
    if (ok && !ctx->reads.empty()) {
        ok = yantra_flush(ctx, s);
    }
    ctx->reads.clear();
    yantra_release_all(ctx);
    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static const char * ggml_backend_yantra_get_name(ggml_backend_t backend) {
    return "Yantra";
    GGML_UNUSED(backend);
}

static void ggml_backend_yantra_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_yantra_context *) backend->context;
    // GGML_YANTRA_STATS=1: the session's counters (round trips, bytes) on stderr.
    const char * stats = std::getenv("GGML_YANTRA_STATS");
    if (stats && *stats && *stats != '0' && g_session) {
        char * js = yantra_session_stats_json(g_session);
        if (js) {
            GGML_LOG_INFO("yantra stats: %s\n", js);
            yantra_string_free(js);
        }
    }
    if (g_session) {
        for (auto & [size, ids] : ctx->pool) {
            for (uint64_t id : ids) {
                yantra_session_free(g_session, id);
            }
        }
    }
    delete ctx;
    delete backend;
}

static ggml_backend_i ggml_backend_yantra_i = {
    /* .get_name                = */ ggml_backend_yantra_get_name,
    /* .free                    = */ ggml_backend_yantra_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_yantra_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_yantra_guid() {
    static ggml_guid guid = { 0x79, 0x61, 0x6e, 0x74, 0x72, 0x61, 0x2d, 0x32,
                              0x31, 0x31, 0x9d, 0x4e, 0x0b, 0x61, 0xc2, 0x5a };
    return &guid;
}

ggml_backend_t ggml_backend_yantra_init(void) {
    if (!yantra_enabled() || yantra_session() == nullptr) {
        return nullptr;
    }
    return new ggml_backend{
        /* .guid    = */ ggml_backend_yantra_guid(),
        /* .iface   = */ ggml_backend_yantra_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_yantra_reg(), 0),
        /* .context = */ new ggml_backend_yantra_context,
    };
}

bool ggml_backend_is_yantra(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_yantra_guid());
}

// ---------------------------------------------------------------------------
// Device.
// ---------------------------------------------------------------------------

static const char * ggml_backend_yantra_device_get_name(ggml_backend_dev_t dev) {
    return "Yantra0";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_yantra_device_get_description(ggml_backend_dev_t dev) {
    const char * v = std::getenv("YANTRA_DEVICE");
    return (v && std::strncmp(v, "remote:", 7) == 0) ? "yantra (remote node)" : "yantra (local)";
    GGML_UNUSED(dev);
}

static void ggml_backend_yantra_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // Buffers are host memory; the node's limit is what bounds residency.
    // GGML_YANTRA_MEMORY overrides the 8 GiB a local node may hold.
    size_t bytes = (size_t) 8 << 30;
    if (const char * v = std::getenv("GGML_YANTRA_MEMORY")) {
        bytes = (size_t) std::strtoull(v, nullptr, 10);
    }
    *free = bytes;
    *total = bytes;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_yantra_device_get_type(ggml_backend_dev_t dev) {
    // GPU, not ACCEL: layers offloaded to it put their weights in its buffer
    // type, which is what lets a weight be uploaded once and stay resident.
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_yantra_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_yantra_device_get_name(dev);
    props->description = ggml_backend_yantra_device_get_description(dev);
    props->type = ggml_backend_yantra_device_get_type(dev);
    ggml_backend_yantra_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_yantra_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_yantra_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static const ggml_backend_buffer_type_i ggml_backend_yantra_buffer_type_i = {
    /* .get_name         = */ ggml_backend_yantra_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_yantra_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ ggml_backend_yantra_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ NULL,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ ggml_backend_yantra_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_yantra_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_yantra_buffer_type_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_yantra_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

static ggml_backend_buffer_type_t ggml_backend_yantra_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_yantra_buffer_type();
    GGML_UNUSED(dev);
}

static int64_t yantra_qgemv_max_rows() {
    static const int64_t rows = [] {
        const char * v = std::getenv("GGML_YANTRA_QGEMV_ROWS");
        return v && *v ? (int64_t) std::strtoll(v, nullptr, 10) : (int64_t) YANTRA_QGEMV_MAX_ROWS;
    }();
    return rows;
}

static bool ggml_backend_yantra_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            {
                const ggml_tensor * src0 = op->src[0];
                const ggml_tensor * src1 = op->src[1];
                // 2-D, contiguous, f32 activations and result; no broadcast.
                if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    return false;
                }
                if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
                    return false;
                }
                if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                    return false;
                }
                switch (src0->type) {
                    case GGML_TYPE_F32:
                        return true;
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                        // ggml's blocks are yantra's QGemv layout (32 weights a block).
                        return src0->ne[0] % 32 == 0 && src1->ne[1] <= yantra_qgemv_max_rows();
                    default:
                        return false;
                }
            }
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_yantra_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_yantra(buft);
    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_yantra_device_i = {
    /* .get_name             = */ ggml_backend_yantra_device_get_name,
    /* .get_description      = */ ggml_backend_yantra_device_get_description,
    /* .get_memory           = */ ggml_backend_yantra_device_get_memory,
    /* .get_type             = */ ggml_backend_yantra_device_get_type,
    /* .get_props            = */ ggml_backend_yantra_device_get_props,
    /* .init_backend         = */ ggml_backend_yantra_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_yantra_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_yantra_device_supports_op,
    /* .supports_buft        = */ ggml_backend_yantra_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// Registry: one device when YANTRA_DEVICE is set, none otherwise.
// ---------------------------------------------------------------------------

static const char * ggml_backend_yantra_reg_get_name(ggml_backend_reg_t reg) {
    return "Yantra";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_yantra_reg_get_device_count(ggml_backend_reg_t reg) {
    return yantra_enabled() ? 1 : 0;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_yantra_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_yantra_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

static const ggml_backend_reg_i ggml_backend_yantra_reg_i = {
    /* .get_name         = */ ggml_backend_yantra_reg_get_name,
    /* .get_device_count = */ ggml_backend_yantra_reg_get_device_count,
    /* .get_device       = */ ggml_backend_yantra_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_yantra_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_yantra_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_yantra_reg)
