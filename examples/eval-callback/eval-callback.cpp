#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdlib>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // LLAMA_CB_FILTER: comma-separated regex patterns matched against the tensor
    // name (same convention as common_debug_cb_user_data's own filters). Lets a
    // targeted trace of a few named tensors run in ONE pass instead of dumping
    // every graph tensor — on a large MoE model that is the difference between a
    // practical whole-model per-layer trace and an impractical one, because the
    // prohibitive cost is the PRINTING, not the compute.
    std::vector<std::string> filters;
    if (const char * f = std::getenv("LLAMA_CB_FILTER")) {
        std::string s(f);
        size_t pos = 0;
        while (true) {
            size_t comma = s.find(',', pos);
            std::string pat = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!pat.empty()) {
                filters.push_back(pat);
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
    }
    common_debug_cb_user_data cb_data(params, filters, false);

    llama_backend_init();
    llama_numa_init(params.numa);

    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
