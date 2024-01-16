#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Fill.hpp>
#include <popops/Loop.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

// Helpers

poplar::Device attach() {
    auto manager = poplar::DeviceManager::createDeviceManager();
    for (auto&& device : manager.getDevices(poplar::TargetType::IPU, 1)) {
        if (device.attach()) {
            return std::move(device);
        }
    }
    assert(false && "could not attach to an IPU");
}

namespace util {  // Graph construction utilities

void print(const std::string& title,
           const poplar::Tensor& tensor,
           poplar::program::Sequence& prog) {
    prog.add(poplar::program::PrintTensor(title, tensor));
}

poplar::Tensor arange(poplar::Graph& graph,
                      const std::optional<poplar::Tensor>& start,
                      unsigned n,
                      unsigned step,
                      poplar::program::Sequence& prog,
                      const poplar::DebugContext& di) {
    auto result =
        graph.addVariable(poplar::UNSIGNED_INT, {n}, poplar::VariableMappingMethod::LINEAR, di);
    popops::iota(graph, result, 0u, prog, di);
    if (step != 1) {
        popops::mulInPlace(graph, result, step, prog, di);
    }
    if (start) {
        popops::addInPlace(graph, result, *start, prog, di);
    }
    return result;
}

poplar::Tensor randn(poplar::Graph& graph,
                     poplar::Type dtype,
                     const std::vector<size_t>& shape,
                     poplar::program::Sequence& prog,
                     const poplar::DebugContext& di) {
    auto dummy = graph.addVariable(dtype, shape, poplar::VariableMappingMethod::LINEAR, di);
    return poprand::normal(graph, nullptr, 0u, dummy, dummy.elementType(), 0.0, 1.0, prog, di);
}

void randnRemote(poplar::Graph& graph,
                 const poplar::RemoteBuffer& buffer,
                 unsigned chunkSize,
                 poplar::program::Sequence& prog,
                 const poplar::DebugContext& di) {
    prog.add(popops::countedLoop(
        graph, 0u, buffer.getRepeats(), chunkSize,
        [&](const poplar::Tensor& startIndex) {
            poplar::program::Sequence prog;
            // indices: [startIndex, startIndex + chunkSize)
            auto indices = arange(graph, startIndex, chunkSize, 1u, prog, {di, "indices"});
            auto data = randn(graph, buffer.elementType(), {chunkSize, buffer.numElements()}, prog,
                              {di, "data"});
            prog.add(poplar::program::Copy(data, buffer, indices, {di, "copy"}));
            return prog;
        },
        di));
}

poplar::Tensor gather(poplar::Graph& graph,
                      const poplar::RemoteBuffer& src,
                      const poplar::Tensor& indices,
                      poplar::program::Sequence& prog,
                      const poplar::DebugContext& di) {
    auto shape = indices.shape();
    shape.push_back(src.numElements());
    auto result =
        graph.addVariable(src.elementType(), shape, poplar::VariableMappingMethod::LINEAR, di);
    prog.add(poplar::program::Copy(src, result.reshape({indices.numElements(), src.numElements()}),
                                   indices.flatten(), di));
    return result;
}

// Attention
//   q: (B, 1, D)
//   k: (B, L, D)
//   v: (B, L, D)
//   returns: (B, 1, D)
poplar::Tensor attnLocal(poplar::Graph& graph,
                         const poplar::Tensor& q,
                         const poplar::Tensor& k,
                         const poplar::Tensor& v,
                         poplar::program::Sequence& prog,
                         const poplar::DebugContext& di) {
    if (q.rank() != 3 || q.dim(1) != 1 || q.dim(0) != k.dim(0) || q.dim(2) != k.dim(2) ||
        k.shapeToString() != v.shapeToString()) {
        std::ostringstream msg;
        msg << "Bad shapes to util::attnLocal(): q.shape=" << q.shapeToString()
            << ", k.shape=" << k.shapeToString() << ", v.shape=" << v.shapeToString();
        throw std::invalid_argument(msg.str());
    }
    auto dtype = q.elementType();
    auto a = poplin::matMulGrouped(graph, q, k.dimShuffle({0, 2, 1}), prog, dtype, {di, "qk"});
    popops::mulInPlace(graph, a, 1 / std::sqrt(q.dim(2)), prog, {di, "a/=sqrt(headDim)"});
    popnn::softmaxStableInPlace(graph, a, prog, {di, "softmax(a)"});
    return poplin::matMulGrouped(graph, a, v, prog, dtype, {di, "kv"});
}

// Attention from remote memory
//   q: (B, 1, D)
//   k: (B*L, D)
//   v: (B*L, D)
//   returns: (B, 1, D)
poplar::Tensor attnRemote(poplar::Graph& graph,
                          const poplar::Tensor& q,
                          const poplar::RemoteBuffer& k,
                          const poplar::RemoteBuffer& v,
                          unsigned chunkSize,  // chunked on sequence axis
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& di) {
    if (q.rank() != 3 || q.dim(1) != 1 || q.dim(2) != k.numElements() ||
        k.getRepeats() % q.dim(0) != 0 || k.getRepeats() != v.getRepeats()) {
        std::ostringstream msg;
        msg << "Bad shapes to util::attnRemote(): q.shape=" << q.shapeToString() << ", k.shape={"
            << k.getRepeats() << "," << k.numElements() << "}, v.shape={" << v.getRepeats() << ","
            << v.numElements() << "}";
        throw std::invalid_argument(msg.str());
    }

    auto headBatchSize = q.dim(0);
    auto sequenceLength = k.getRepeats() / q.dim(0);
    auto dtype = q.elementType();

    // Chunked QK product
    auto a = graph.addVariable(dtype, {headBatchSize, 1, sequenceLength},
                               poplar::VariableMappingMethod::LINEAR, {di, "a"});

    auto batchIndices = util::arange(graph, std::nullopt, headBatchSize, sequenceLength, prog,
                                     {di, "batchIndices"});
    prog.add(popops::countedLoop(
        graph, 0u, sequenceLength, chunkSize,
        [&](const poplar::Tensor& startIndex) {
            poplar::program::Sequence prog;

            // Gather
            auto indices = util::arange(graph, startIndex, chunkSize, 1u, prog, {di, "indices"});
            indices = popops::add(graph, batchIndices.reshape({headBatchSize, 1}),
                                  indices.reshape({1, chunkSize}), prog, {di, "indices"});
            auto kChunk =
                util::gather(graph, k, indices, prog, {di, "gather_h"});  // (batch, chunk, headDim)

            // Matmul
            auto qk = poplin::matMulGrouped(graph, q, kChunk.dimShuffle({0, 2, 1}), prog, dtype,
                                            {di, "qk"});

            // Copy
            poplar::program::Switch copyToA(startIndex.reshape({}), {di, "a[chunk]=qk"});
            for (auto i = 0u; i < sequenceLength; i += chunkSize) {
                copyToA.add(i, poplar::program::Copy(qk, a.slice({i, i + chunkSize}, 2u),
                                                     /*dontOutline*/ false, {di, "chunkCopy"}));
            }
            prog.add(copyToA);

            return prog;
        },
        {di, "loop_qk"}));

    popops::mulInPlace(graph, a, 1 / std::sqrt(q.dim(2)), prog, {di, "a/=sqrt(headDim)"});
    popnn::softmaxStableInPlace(graph, a, prog, {di, "softmax(a)"});

    // Chunked AV product
    auto y = graph.clone(q);
    popops::fill(graph, y, prog, 0.0f, {di, "y=0"});
    prog.add(popops::countedLoop(
        graph, 0u, sequenceLength, chunkSize,
        [&](const poplar::Tensor& startIndex) {
            poplar::program::Sequence prog;

            // Copy
            auto aChunk = graph.clone(a.slice({0, chunkSize}, 2u));
            poplar::program::Switch copyFromA(startIndex.reshape({}), {di, "a[chunk]"});
            for (auto i = 0u; i < sequenceLength; i += chunkSize) {
                copyFromA.add(i, poplar::program::Copy(a.slice({i, i + chunkSize}, 2u), aChunk,
                                                       /*dontOutline*/ false, {di, "chunkCopy"}));
            }
            prog.add(copyFromA);

            // Gather
            auto indices = util::arange(graph, startIndex, chunkSize, 1u, prog, {di, "indices"});
            indices = popops::add(graph, batchIndices.reshape({headBatchSize, 1}),
                                  indices.reshape({1, chunkSize}), prog, {di, "indices"});
            auto vChunk =
                util::gather(graph, v, indices, prog, {di, "gather_v"});  // (batch, chunk, headDim)

            // Matmul
            auto kv = poplin::matMulGrouped(graph, aChunk, vChunk, prog, dtype, {di, "kv"});

            // Accumulate
            popops::addInPlace(graph, y, kv, prog, {di, "y += kv"});

            return prog;
        },
        {di, "loop_kv"}));

    return y;
}

}  // namespace util

// Core benchmark

namespace benchmark {

struct Config {
    // Problem
    unsigned batchSize;
    unsigned nHead;
    unsigned headDim;
    unsigned sequenceLength;
    poplar::Type dtype;

    // Execution
    std::string method;
    unsigned chunkSize;

    unsigned headBatchSize() const { return batchSize * nHead; }
};

struct Base {
    poplar::Graph graph;
    Config c;

    Base(poplar::Graph&& graph_, const Config& c, const std::string& name)
        : graph(std::move(graph_)), c(c) {
        assert(name == c.method);
        popops::addCodelets(graph);
        poprand::addCodelets(graph);
        poplin::addCodelets(graph);
    }
    virtual ~Base() {}

    virtual poplar::program::Sequence prepare() = 0;
    virtual poplar::program::Sequence run() = 0;
};

struct AttnLocal : Base {
    poplar::Tensor k;
    poplar::Tensor v;

    AttnLocal(poplar::Graph&& graph_, const Config& c) : Base(std::move(graph_), c, "attn-local") {
        if (c.chunkSize != 0) {
            throw std::invalid_argument("Config::chunkSize should be 0 for attn-local");
        }
        k = graph.addVariable(c.dtype, {c.headBatchSize(), c.sequenceLength, c.headDim},
                              poplar::VariableMappingMethod::LINEAR, "k");
        v = graph.addVariable(c.dtype, {c.headBatchSize(), c.sequenceLength, c.headDim},
                              poplar::VariableMappingMethod::LINEAR, "v");
    }

    poplar::program::Sequence prepare() {
        poplar::DebugContext di("prepare");
        poplar::program::Sequence prog;
        prog.add(poplar::program::Copy(util::randn(graph, c.dtype, k.shape(), prog, {di, "k"}), k,
                                       /*dontOutline*/ false, {di, "k"}));
        prog.add(poplar::program::Copy(util::randn(graph, c.dtype, v.shape(), prog, {di, "v"}), v,
                                       /*dontOutline*/ false, {di, "v"}));
        return prog;
    }

    poplar::program::Sequence run() {
        poplar::DebugContext di("run");
        poplar::program::Sequence prog;
        auto q = util::randn(graph, c.dtype, {c.headBatchSize(), 1, c.headDim}, prog, {di, "q"});
        auto y = util::attnLocal(graph, q, k, v, prog, di);
        // util::print("y", y, prog);
        return prog;
    }
};

struct AttnRemote : Base {
    // poplar::RemoteBuffer bufferK1;
    poplar::RemoteBuffer bufferK;
    poplar::RemoteBuffer bufferV;

    AttnRemote(poplar::Graph&& graph_, const Config& c)
        : Base(std::move(graph_), c, "attn-remote") {
        if (c.sequenceLength % c.chunkSize != 0) {
            throw std::invalid_argument("`sequenceLength` must be a multiple of `chunkSize`");
        }
        // bufferK1 =
        //     graph.addRemoteBuffer("K1", c.dtype, c.sequenceLength, c.headBatchSize() *
        //     c.headDim);
        bufferK =
            graph.addRemoteBuffer("K", c.dtype, c.headDim, c.headBatchSize() * c.sequenceLength);
        bufferV =
            graph.addRemoteBuffer("V", c.dtype, c.headDim, c.headBatchSize() * c.sequenceLength);
    }

    poplar::program::Sequence prepare() {
        poplar::DebugContext di("prepare");
        poplar::program::Sequence prog;
        // util::randnRemote(graph, bufferK1, c.headDim, prog, {di, "K1"});
        util::randnRemote(graph, bufferK, c.headBatchSize() * c.chunkSize, prog, {di, "k"});
        util::randnRemote(graph, bufferV, c.headBatchSize() * c.chunkSize, prog, {di, "v"});
        return prog;
    }

    poplar::program::Sequence run() {
        poplar::DebugContext di("run");
        poplar::program::Sequence prog;
        auto q = util::randn(graph, c.dtype, {c.headBatchSize(), 1, c.headDim}, prog, {di, "q"});
        auto y = util::attnRemote(graph, q, bufferK, bufferV, c.chunkSize, prog, di);
        // util::print("y", y, prog);
        return prog;
    }
};

Base* get(poplar::Graph&& graph_, const Config& c) {
    if (c.method == "attn-local") {
        return new AttnLocal(std::move(graph_), c);
    }
    if (c.method == "attn-remote") {
        return new AttnRemote(std::move(graph_), c);
    }
    std::ostringstream msg;
    msg << "Unknown Config::method: " << c.method;
    throw std::invalid_argument(msg.str());
}

}  // namespace benchmark

int main() {
    auto device = attach();

    // Test config
    benchmark::Config config{
        /*batchSize*/ 2u,
        /*nHead*/ 2u,
        /*headDim*/ 8u,
        /*sequenceLength*/ 10u,
        /*dtype*/ poplar::HALF,
        // /*method*/ "attn-local",
        // /*chunkSize*/ 0u
        /*method*/ "attn-remote",
        /*chunkSize*/ 5u
    };
    // Full configs
    // benchmark::Config config{/*batchSize*/ 1u,
    //                          /*nHead*/ 32u,
    //                          /*headDim*/ 128u,
    //                          /*sequenceLength*/ 4096u,
    //                          /*dtype*/ poplar::HALF,
    //                          /*method*/ "attn-local",
    //                          /*chunkSize*/ 0u};
    // benchmark::Config config{/*batchSize*/ 16u,
    //                          /*nHead*/ 32u,
    //                          /*headDim*/ 128u,
    //                          /*sequenceLength*/ 4096u,
    //                          /*dtype*/ poplar::HALF,
    //                          /*method*/ "attn-remote",
    //                          /*chunkSize*/ 128u};

    std::shared_ptr<benchmark::Base> builder(benchmark::get({device}, config));
    auto t0 = std::chrono::system_clock::now();
    auto showElapsed = [&t0](const std::string& name) {
        auto t1 = std::chrono::system_clock::now();
        std::cerr << "[" << name << "] "
                  << std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0).count()
                  << std::endl;
        t0 = t1;
    };
    poplar::Engine engine(builder->graph,
                          {builder->prepare(), builder->run(), poplar::program::Sequence()});
    showElapsed("compile");
    engine.load(device);
    showElapsed("load");
    for (auto i = 0u; i < 5u; ++i) {
        engine.run(2);
        showElapsed("empty");
    }
    engine.run(0);
    showElapsed("prepare");
    for (auto i = 0u; i < 5u; ++i) {
        engine.run(1);
        showElapsed("run");
    }
    return 0;
}
