#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <graphcore_target_access/IPUAttributes.h>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Fill.hpp>
#include <popops/Loop.hpp>
#include <popops/Reduce.hpp>
#include <popops/TopK.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

// Helpers

poplar::Device attach(unsigned count) {
    auto manager = poplar::DeviceManager::createDeviceManager();
    for (auto&& device : manager.getDevices(poplar::TargetType::IPU, count)) {
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

// Grouped gather from remote memory
//   src: (G*N, D)
//   indices: (G, K)
//   returns: (G, K, D)
poplar::Tensor gatherGrouped(poplar::Graph& graph,
                             const poplar::RemoteBuffer& src,
                             const poplar::Tensor& indices,
                             poplar::program::Sequence& prog,
                             const poplar::DebugContext& di) {
    if (src.getRepeats() % indices.dim(0) != 0) {
        throw std::invalid_argument(
            "gatherGrouped `src` repeat count is not a multiple of the number of groups"
            " implied by indices.dim(0)");
    }
    if (indices.rank() != 2) {
        throw std::invalid_argument("gatherGrouped expected indices.rank == 2");
    }
    auto groupOffsets = arange(graph, std::nullopt, indices.dim(0),
                               src.getRepeats() / indices.dim(0), prog, {di, "groupOffsets"});
    auto flatIndices =  // (G, K)
        popops::add(graph, groupOffsets.expand({1}), indices, prog, {di, "flatIndices"});
    return gather(graph, src, flatIndices, prog, di);  // (G, K, D)
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
    prog.add(popops::countedLoop(
        graph, 0u, sequenceLength, chunkSize,
        [&](const poplar::Tensor& startIndex) {
            poplar::program::Sequence prog;

            // Gather
            auto kChunk = util::gatherGrouped(
                graph, k,
                util::arange(graph, startIndex, chunkSize, 1u, prog, {di, "indices"})
                    .expand({0})
                    .broadcast(headBatchSize, 0),
                prog, {di, "gather_k"});

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
            auto vChunk = util::gatherGrouped(
                graph, v,
                util::arange(graph, startIndex, chunkSize, 1u, prog, {di, "indices"})
                    .expand({0})
                    .broadcast(headBatchSize, 0),
                prog, {di, "gather_v"});

            // Matmul
            auto kv = poplin::matMulGrouped(graph, aChunk, vChunk, prog, dtype, {di, "kv"});

            // Accumulate
            popops::addInPlace(graph, y, kv, prog, {di, "y += kv"});

            return prog;
        },
        {di, "loop_kv"}));

    return y;
}

poplar::Tensor sparqAttn(poplar::Graph& graph,
                         const poplar::Tensor& Q,
                         const poplar::RemoteBuffer& K1,
                         const poplar::RemoteBuffer& K2,
                         const poplar::RemoteBuffer& V,
                         const poplar::RemoteBuffer& Vmean,
                         unsigned k1,
                         unsigned k2,
                         poplar::program::Sequence& prog,
                         const poplar::DebugContext& di) {
    if (Q.rank() != 3 || Q.dim(1) != 1 || Q.dim(2) != K2.numElements() ||
        K2.getRepeats() % Q.dim(0) != 0 || K2.getRepeats() != V.getRepeats() ||
        // SparQ-specific
        K1.numElements() != K2.getRepeats() / Q.dim(0) || K1.getRepeats() != Q.dim(0) * Q.dim(2)) {
        std::ostringstream msg;
        msg << "Bad shapes to util::sparqAttn(): Q.shape=" << Q.shapeToString()     //
            << ", K1.shape={" << K1.getRepeats() << "," << K1.numElements() << "}"  //
            << ", K2.shape={" << K2.getRepeats() << "," << K2.numElements() << "}"  //
            << ", v.shape={" << V.getRepeats() << "," << V.numElements() << "}";
        throw std::invalid_argument(msg.str());
    }

    // auto headBatchSize = Q.dim(0);
    // auto sequenceLength = K1.numElements();
    auto dtype = Q.elementType();

    // Step 1
    auto absQ = popops::abs(graph, Q, prog, {di, "abs(Q)"});
    auto topk1 = popops::topKWithPermutation(
        graph, prog, absQ, popops::TopKParams(k1, /*largest*/ true, popops::SortOrder::NONE));
    auto plan = popops::embedding::plan(graph, Q.elementType(), Q.dim(0), Q.dim(2), 1u, {k1}, {});
    auto Qhat = popops::groupedMultiSlice(graph, Q.dimShuffle({0, 2, 1}),
                                          topk1.second.squeeze({1}).expand({2}), {0}, {1}, prog,
                                          plan, {}, {di, "Q[i1]"})
                    .squeeze({2, 3})
                    .expand({1});
    auto Khat = gatherGrouped(graph, K1, topk1.second.squeeze({1}), prog, {di, "K[i1]"});
    auto shat = poplin::matMulGrouped(graph, Qhat, Khat, prog, dtype, {di, "Q[i1] @ K[i1]"});
    namespace pe = popops::expr;
    auto scale = popops::map(
        graph, pe::Sqrt(pe::Const(Q.dim(2)) * pe::_1 / pe::_2),
        {popops::reduce(graph, topk1.first, {2}, popops::Operation::ADD, prog,
                        {di, "sum(abs(Q)[i1])"}),
         popops::reduce(graph, absQ, {2}, popops::Operation::ADD, prog, {di, "sum(abs(Q))"})},
        prog, {di, "scale"});
    popops::divInPlace(graph, shat, scale.expand({2}), prog, {di, "shat/scale"});
    popnn::softmaxStableInPlace(graph, shat, prog, {di, "shat"});

    // Step 2
    auto topk2 = popops::topKWithPermutation(
        graph, prog, shat, popops::TopKParams(k2, /*largest*/ true, popops::SortOrder::NONE));
    auto y_ = attnLocal(graph, Q,
                        gatherGrouped(graph, K2, topk2.second.squeeze({1}), prog, {di, "K[i2]"}),
                        gatherGrouped(graph, V, topk2.second.squeeze({1}), prog, {di, "V[i2]"}),
                        prog, {di, "attn"});

    // Step 3
    auto VmeanLocal = graph.clone(y_);
    prog.add(poplar::program::Copy(Vmean, VmeanLocal, "VMean"));
    auto alpha = popops::reduce(graph, topk2.first, {2}, popops::Operation::ADD, prog,
                                {di, "sum(shat[i2])"});
    return popops::map(graph, pe::_3 * pe::_1 + (1 - pe::_3) * pe::_2,
                       {y_, VmeanLocal, alpha.expand({2})}, prog, {di, "lerp"});
}

}  // namespace util

// Benchmarking adapters

namespace benchmark {

struct Config {
    // Problem
    unsigned batchSize;
    unsigned nHead;
    unsigned headDim;
    unsigned sequenceLength;
    poplar::Type dtype;

    // Technique
    std::string kernel;
    unsigned chunkSize;
    unsigned k1;
    unsigned k2;

    // Benchmarking
    unsigned reps;
    bool showResult;

    unsigned headBatchSize() const { return batchSize * nHead; }
};

struct Base {
    poplar::Graph graph;
    Config c;

    Base(poplar::Graph&& graph_, const Config& c, const std::string& name)
        : graph(std::move(graph_)), c(c) {
        assert(name == c.kernel);
        popops::addCodelets(graph);
        poprand::addCodelets(graph);
        poplin::addCodelets(graph);
    }
    virtual ~Base() {}

    virtual poplar::program::Sequence prepare() = 0;

    poplar::program::Sequence run() {
        poplar::DebugContext di("run");
        poplar::program::Sequence prog;
        auto Q = util::randn(graph, c.dtype, {c.headBatchSize(), 1, c.headDim}, prog, {di, "Q"});
        poplar::program::Sequence loopBody;
        auto Y = runSingle(Q, loopBody, di);
        prog.add(poplar::program::Repeat(c.reps, loopBody, di));
        if (c.showResult) {
            util::print("Y", Y.squeeze({1}), prog);
        }
        return prog;
    }
    virtual poplar::Tensor runSingle(const poplar::Tensor& q,
                                     poplar::program::Sequence& prog,
                                     const poplar::DebugContext& di) = 0;
};

struct AttnLocal : Base {
    poplar::Tensor K;
    poplar::Tensor V;

    AttnLocal(poplar::Graph&& graph_, const Config& c) : Base(std::move(graph_), c, "attn-local") {
        if (c.chunkSize != 0) {
            throw std::invalid_argument("Config::chunkSize should be 0 for attn-local");
        }
        K = graph.addVariable(c.dtype, {c.headBatchSize(), c.sequenceLength, c.headDim},
                              poplar::VariableMappingMethod::LINEAR, "K");
        V = graph
                .addVariable(c.dtype, {c.headBatchSize(), c.headDim, c.sequenceLength},
                             poplar::VariableMappingMethod::LINEAR, "V")
                .dimShuffle({0, 2, 1});
    }

    poplar::program::Sequence prepare() {
        poplar::DebugContext di("prepare");
        poplar::program::Sequence prog;
        prog.add(poplar::program::Copy(util::randn(graph, c.dtype, K.shape(), prog, {di, "K"}), K,
                                       /*dontOutline*/ false, {di, "K"}));
        prog.add(poplar::program::Copy(util::randn(graph, c.dtype, V.shape(), prog, {di, "V"}), V,
                                       /*dontOutline*/ false, {di, "V"}));
        return prog;
    }

    poplar::Tensor runSingle(const poplar::Tensor& Q,
                             poplar::program::Sequence& prog,
                             const poplar::DebugContext& di) {
        return util::attnLocal(graph, Q, K, V, prog, di);
    }
};

struct AttnRemote : Base {
    poplar::RemoteBuffer K;
    poplar::RemoteBuffer V;

    AttnRemote(poplar::Graph&& graph_, const Config& c)
        : Base(std::move(graph_), c, "attn-remote") {
        if (c.chunkSize == 0 || c.sequenceLength % c.chunkSize != 0) {
            throw std::invalid_argument("`sequenceLength` must be a multiple of `chunkSize`");
        }
        K = graph.addRemoteBuffer("K", c.dtype, c.headDim, c.headBatchSize() * c.sequenceLength);
        V = graph.addRemoteBuffer("V", c.dtype, c.headDim, c.headBatchSize() * c.sequenceLength);
    }

    poplar::program::Sequence prepare() {
        poplar::DebugContext di("prepare");
        poplar::program::Sequence prog;
        util::randnRemote(graph, K, c.headBatchSize() * c.chunkSize, prog, {di, "K"});
        util::randnRemote(graph, V, c.headBatchSize() * c.chunkSize, prog, {di, "V"});
        return prog;
    }

    poplar::Tensor runSingle(const poplar::Tensor& Q,
                             poplar::program::Sequence& prog,
                             const poplar::DebugContext& di) {
        return util::attnRemote(graph, Q, K, V, c.chunkSize, prog, di);
    }
};

struct SparQAttn : Base {
    poplar::RemoteBuffer K1;
    poplar::RemoteBuffer K2;
    poplar::RemoteBuffer V;
    poplar::RemoteBuffer Vmean;

    SparQAttn(poplar::Graph&& graph_, const Config& c) : Base(std::move(graph_), c, "sparq-attn") {
        if (c.chunkSize != 0) {
            throw std::invalid_argument("Config::chunkSize should be 0 for sparq-attn");
        }
        K1 = graph.addRemoteBuffer("K1", c.dtype, c.sequenceLength, c.headBatchSize() * c.headDim);
        K2 = graph.addRemoteBuffer("K2", c.dtype, c.headDim, c.headBatchSize() * c.sequenceLength);
        V = graph.addRemoteBuffer("V", c.dtype, c.headDim, c.headBatchSize() * c.sequenceLength);
        Vmean = graph.addRemoteBuffer("Vmean", c.dtype, c.headBatchSize() * c.headDim);
    }

    poplar::program::Sequence prepare() {
        poplar::DebugContext di("prepare");
        poplar::program::Sequence prog;
        util::randnRemote(graph, K1, c.headBatchSize() * c.k1, prog, {di, "K1"});
        util::randnRemote(graph, K2, c.headBatchSize() * c.k2, prog, {di, "K2"});
        util::randnRemote(graph, V, c.headBatchSize() * c.k2, prog, {di, "V"});
        prog.add(poplar::program::Copy(
            util::randn(graph, c.dtype, {Vmean.numElements()}, prog, {di, "Vmean"}), Vmean,
            {di, "Vmean"}));
        return prog;
    }

    poplar::Tensor runSingle(const poplar::Tensor& Q,
                             poplar::program::Sequence& prog,
                             const poplar::DebugContext& di) {
        return util::sparqAttn(graph, Q, K1, K2, V, Vmean, c.k1, c.k2, prog, di);
    }
};

Base* get(poplar::Graph&& graph_, const Config& c) {
    if (c.kernel == "attn-local") {
        return new AttnLocal(std::move(graph_), c);
    }
    if (c.kernel == "attn-remote") {
        return new AttnRemote(std::move(graph_), c);
    }
    if (c.kernel == "sparq-attn") {
        return new SparQAttn(std::move(graph_), c);
    }
    std::ostringstream msg;
    msg << "Unknown Config::kernel: " << c.kernel;
    throw std::invalid_argument(msg.str());
}

}  // namespace benchmark

namespace {

template <class T>
T get(const poplar::OptionFlags& options, const std::string& key) {
    T result;
    auto it = options.find(key);
    if (it == options.end()) {
        std::ostringstream msg;
        msg << "Couldn't find option key \"" << key << "\"";
        throw std::invalid_argument(msg.str());
    }
    std::istringstream in(it->second);
    in >> result;
    if (!in.eof()) {
        std::ostringstream msg;
        msg << "Couldn't parse option {\"" << key << "\": \"" << options.at(key) << "\"}";
        throw std::invalid_argument(msg.str());
    }
    return result;
}

poplar::Type toDtype(const std::string& name) {
    if (name == "float32") {
        return poplar::FLOAT;
    }
    if (name == "float16") {
        return poplar::HALF;
    }
    throw std::invalid_argument("Couldn't parse dtype, expected \"float32\" or \"float16\"");
}

struct Stopwatch {
    typedef std::chrono::system_clock clock;
    clock::time_point start;

    Stopwatch() : start(clock::now()) {}

    float lap() {
        auto current = clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<float>>(current - start).count();
        start = current;
        return elapsed;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./sparq_benchmark '{\"option\": value}'" << std::endl;
        return 1;
    }
    poplar::OptionFlags opt;
    {
        std::istringstream in(argv[1]);
        poplar::readJSON(in, opt);
    }

    benchmark::Config config{
        // Problem
        get<unsigned>(opt, "batch_size"),
        get<unsigned>(opt, "n_head"),
        get<unsigned>(opt, "head_dim"),
        get<unsigned>(opt, "sequence_length"),
        toDtype(get<std::string>(opt, "dtype")),

        // Method
        get<std::string>(opt, "kernel"),
        get<unsigned>(opt, "chunk_size"),
        get<unsigned>(opt, "k1"),
        get<unsigned>(opt, "k2"),

        // Benchmark
        get<unsigned>(opt, "inner_reps"),
        /*showResult*/ false,
    };
    auto device = attach(get<unsigned>(opt, "n_ipu"));
    unsigned warmup = get<unsigned>(opt, "warmup");
    unsigned reps = get<unsigned>(opt, "reps");

    Stopwatch stopwatch;
    std::shared_ptr<benchmark::Base> builder(benchmark::get(
        {device, poplar::replication_factor(device.getTarget().getNumIPUs())}, config));
    auto durationConstruct = stopwatch.lap();

    poplar::Engine engine(
        builder->graph, {builder->prepare(), builder->run()},
        {{"target.extendedMemory", "false"},  // Enabling this causes a performance regression
         {"target.hostSyncTimeout", "1800"}});
    auto durationCompile = stopwatch.lap();

    engine.load(device);
    auto durationLoad = stopwatch.lap();

    engine.disableExecutionProfiling();
    engine.run(0);
    engine.enableExecutionProfiling();
    auto durationPrepare = stopwatch.lap();

    std::vector<float> durations;
    for (auto i = 0u; i < warmup + reps; ++i) {
        engine.run(1);
        // Record the average time over "inner" reps
        durations.push_back(stopwatch.lap() / config.reps);
    }

    // Print JSON results manually
    std::cout << "{"
              << "\"device_name\":\""
              << device.getAttribute(IPUAttributes::AttributeId::BoardVariant) << "\""
              << ",\"device_clock\":" << device.getTarget().getTileClockFrequency()
              << ",\"duration_construct\":" << durationConstruct  //
              << ",\"duration_compile\":" << durationCompile      //
              << ",\"duration_load\":" << durationLoad            //
              << ",\"duration_prepare\":" << durationPrepare      //
              << ",\"duration\":[";
    for (auto i = 0u; i < reps; ++i) {
        std::cout << (i ? "," : "") << durations.at(warmup + i);
    }
    std::cout << "]}";
    return 0;
}
