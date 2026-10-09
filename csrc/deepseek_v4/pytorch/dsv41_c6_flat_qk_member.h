// SPDX-License-Identifier: Apache-2.0
// Included inside an OpBackend class. QK only: PV precision/layout is unchanged.
std::vector<synapse_helpers::tensor> build_c6_qk(synapse_helpers::graph& graph,
    synTensor query, synTensor keys, int64_t tokens, int64_t heads, int64_t width) {
    synGEMMParams qk{false,true};
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
    const int64_t groups = heads / 8;
    auto q = ReshapeHelper(graph, query, {tokens,groups,512,8}, at::kBFloat16);
    auto k = ReshapeHelper(graph, keys, {tokens,1,width,512}, at::kBFloat16);
    synGEMMParams interleaved_qk{true,true};
    auto result = BuildNode(this, graph, {"batch_gemm", {q.get(),k.get()},
        {{{tokens,groups,8,width},at::kFloat}}, &interleaved_qk, sizeof(interleaved_qk)});
    std::vector<synapse_helpers::tensor> output;
    output.push_back(ReshapeHelper(graph,result[0].get(),{tokens,heads,width},at::kFloat));
    return output;
#endif
#if defined(DSV41_C6_FLAT_QK) && DSV41_C6_FLAT_QK
    if(tokens>=2 && tokens<=6) {
        auto q=ReshapeHelper(graph,query,{tokens*heads,512},at::kBFloat16);
        auto k=ReshapeHelper(graph,keys,{tokens*width,512},at::kBFloat16);
        auto full=BuildNode(this,graph,{"gemm",{q.get(),k.get()},
            {{{tokens*heads,tokens*width},at::kFloat}},&qk,sizeof(qk)});
#if defined(DSV41_C6_FLAT_QK_DIRECT) && DSV41_C6_FLAT_QK_DIRECT
        std::vector<synapse_helpers::tensor> result;
        result.push_back(ReshapeHelper(graph, full[0].get(), {tokens,heads,tokens*width}, at::kFloat));
        return result;
#else
        std::vector<synapse_helpers::tensor> slices;
        std::vector<synTensor> inputs;
        slices.reserve(tokens*2);inputs.reserve(tokens);
        for(int64_t row=0;row<tokens;++row) {
            synSliceParams slice{};
            for(unsigned d=0;d<sizeof(slice.axes)/sizeof(slice.axes[0]);++d) {
                slice.axes[d]=d;slice.steps[d]=1;slice.ends[d]=1;
            }
            slice.starts[0]=row*width;slice.ends[0]=(row+1)*width;
            slice.starts[1]=row*heads;slice.ends[1]=(row+1)*heads;
            auto part=BuildNode(this,graph,{"slice",{full[0].get()},
                {{{heads,width},at::kFloat}},&slice,sizeof(slice)});
            slices.push_back(std::move(part[0]));
            auto expanded=ReshapeHelper(graph,slices.back().get(),{1,heads,width},at::kFloat);
            inputs.push_back(expanded.get());slices.push_back(std::move(expanded));
        }
        synConcatenateParams concat{};concat.axis=2;
        return BuildNode(this,graph,{"concat",inputs,{{{tokens,heads,width},at::kFloat}},
            &concat,sizeof(concat)});
#endif
    }
#endif
    return BuildNode(this,graph,{"batch_gemm",{query,keys},
        {{{tokens,heads,width},at::kFloat}},&qk,sizeof(qk)});
}
