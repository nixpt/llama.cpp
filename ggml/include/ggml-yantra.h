#pragma once

// ggml-yantra: a backend whose device is yantra (nixpt/yantra#211).
//
// Built only with -DGGML_YANTRA=ON, and inert at run time unless
// YANTRA_DEVICE is set: `local` (an in-process yantra node on the CPU
// driver) or `remote:SPEC` (a rented node, see yantra docs/RENTER.md).
// It runs the ops yantra supports (supports_op); ggml's scheduler runs the
// rest on the CPU backend. Weights are uploaded to yantra once
// (content-addressed, resident) and each graph split is one exchange.
// No llama.cpp code runs on the provider.

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t ggml_backend_yantra_init(void);

GGML_BACKEND_API bool ggml_backend_is_yantra(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_yantra_reg(void);

#ifdef  __cplusplus
}
#endif
