/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "graph_ext_inst_context.hpp"
#include "graph_ext_inst_registry.hpp"

#include "source/opt/pass.h"
#include <exception>
#include <utility>

#include <spirv-tools/optimizer.hpp>

namespace spvtools::opt {

class GraphPassExtInst final : public Pass {
  public:
    GraphPassExtInst(GraphPipeline &_graphPipeline, GraphExtInstImports imports, std::exception_ptr &_error)
        : graphPipeline{_graphPipeline}, imports{std::move(imports)}, error{_error} {}
    ~GraphPassExtInst() override = default;
    const char *name() const override { return "graph-pass-extinst"; }

  private:
    using Decoders = std::unordered_map<uint32_t, GraphExtInstDecoder *>;

    Status Process() override;
    void handleGraphs(GraphExtInstContext &loweringContext, const Decoders &decoders);
    void handleGraph(const Graph *graph, const GraphExtInstContext &loweringContext, const Decoders &decoders);

    GraphPipeline &graphPipeline;
    GraphExtInstImports imports;
    // SPIRV-Tools is built without exceptions, so the pass must not let one unwind through it
    std::exception_ptr &error;
};

} // namespace spvtools::opt

namespace spvtools {

inline Optimizer::PassToken createGraphPass(opt::GraphPipeline &graphPipeline, opt::GraphExtInstImports imports,
                                            std::exception_ptr &error) {
    return Optimizer::PassToken{MakeUnique<opt::GraphPassExtInst>(graphPipeline, std::move(imports), error)};
}

} // namespace spvtools
