// SPDX-License-Identifier: Apache-2.0
// Offline public-API check; no device acquisition and no graph export.
#include "synapse_api.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(synStatus status, const char* operation) {
    if (status != synSuccess) throw std::runtime_error(std::string(operation)+": "+std::to_string(status));
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    synRecipeHandle recipe = nullptr;
    try {
        require(synInitialize(), "initialize");
        bool first = true;size_t readyRecipes = 0,ordinaryRecipes = 0;
        std::cout << "{\"device_acquired\":false,\"recipes\":[";
        for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() != ".recipe") continue;
            require(synRecipeDeSerialize(&recipe, entry.path().c_str()), "deserialize");
            const synRecipeAttribute attribute = RECIPE_ATTRIBUTE_NUM_EXTERNAL_TENSORS;
            uint64_t count = 0;
            require(synRecipeGetAttribute(&count, &attribute, 1, recipe), "external tensor count");
            if (count > 128) throw std::runtime_error("External count exceeds replay plan capacity");
            std::vector<uint64_t> ids(count);
            if (count) require(synTensorExtExtractExecutionOrder(recipe, count, ids.data()), "execution order");
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"path\":\"" << entry.path().string() << "\",\"external_count\":" << count
                      << ",\"ordered_tensor_ids\":[";
            for (size_t i = 0; i < ids.size(); ++i) {
                synRetrievedLaunchTensorInfoExt info{};info.tensorId = ids[i];
                require(synTensorRetrieveLaunchInfoByIdExt(recipe, 1, &info), "external launch tensor");
                if (info.isInput || info.tensorDataType != syn_type_bf16 || info.tensorDims != 2 ||
                    info.tensorMaxSize[0] != 5120 || (info.tensorMaxSize[1] != 2 && info.tensorMaxSize[1] != 6))
                    throw std::runtime_error("Signal does not name the persistent peer packet");
                if (i) std::cout << ',';
                std::cout << ids[i];
            }
            std::cout << "]}";
            readyRecipes += count > 0;ordinaryRecipes += count == 0;
            require(synRecipeDestroy(recipe), "recipe destroy");recipe = nullptr;
        }
        if (readyRecipes != 4 || ordinaryRecipes != 4)
            throw std::runtime_error("Expected four candidate and four reference shape variants");
        std::cout << "],\"passed\":true}\n";
        require(synDestroy(), "destroy");return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (recipe) synRecipeDestroy(recipe);
        synDestroy();return 1;
    }
}
