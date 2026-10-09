// SPDX-License-Identifier: Apache-2.0
// Disabled research capability, compiled against the locked runtime headers.
// Read typed event metadata; never reinterpret event handles or patch offsets.
#include "runtime/common/syn_singleton.hpp"
#include "runtime/scal/common/scal_event.hpp"
#include "synapse_api_types.h"

extern "C" synStatus dsv41ReceiveEventSyncInfo(
    synEventHandle handle, synNativeGraphSyncInfo* info, uint32_t* onHcl) {
    if (!handle || !info || !onHcl || !synSingleton::isSynapseInitialized())
        return synInvalidArgument;
    auto* event = dynamic_cast<ScalEvent*>(synSingleton::getInstanceInternal()->getEventInterface(handle));
    if (!event) return synInvalidArgument;
    std::lock_guard<ScalEvent> lock(*event);
    const uint64_t index = event->isOnHclStream() ? event->hclSyncInfo.long_so_index : event->longSo.m_index;
    const uint64_t target = event->isOnHclStream() ? event->hclSyncInfo.targetValue : event->longSo.m_targetValue;
    if (!target || index > UINT32_MAX)
        return synInvalidArgument;
    *info = {uint32_t(index), target};
    *onHcl = event->isOnHclStream();
    return synSuccess;
}

// Bind an owned, untimed NIC event to one completion within the just-queued
// batch. Recording after a batch normally names its last completion; an
// earlier future consumer must never wait for that last point by accident.
extern "C" synStatus dsv41ReceiveBindBatchEvent(
    synEventHandle handle, const synNativeGraphSyncInfo* info,
    uint64_t beforeBatch, uint64_t batchEnd) {
    if (!handle || !info || !synSingleton::isSynapseInitialized() ||
        beforeBatch >= info->targetValue || info->targetValue > batchEnd ||
        batchEnd >= (uint64_t(1) << 60)) return synInvalidArgument;
    auto* event = dynamic_cast<ScalEvent*>(synSingleton::getInstanceInternal()->getEventInterface(handle));
    if (!event) return synInvalidArgument;
    std::lock_guard<ScalEvent> lock(*event);
    if (event->collectTime || !event->isOnHclStream() || event->isInternalSignalingEvent() ||
        event->hclSyncInfo.long_so_index != info->longSoIndex ||
        event->hclSyncInfo.targetValue != batchEnd) return synInvalidArgument;
    event->hclSyncInfo.targetValue = info->targetValue;
    event->longSo = {info->longSoIndex, info->targetValue};
    event->setWaitMode(EventInterface::unset);
    return synSuccess;
}

// A cold NIC event supplies only the queue/SO namespace. Its old target need
// not be refreshed in the replay callback. This is essential when native
// compute and NIC share a Synapse stream handle: eventRecord there would try
// to reacquire the compute queue lock held by ReplayPlan.
extern "C" synStatus dsv41ReceiveBindEventFromTemplate(
    synEventHandle destination, synEventHandle seed,
    const synNativeGraphSyncInfo* info, uint64_t beforeBatch, uint64_t batchEnd) {
    if (!destination || !seed || !info || !synSingleton::isSynapseInitialized() ||
        beforeBatch >= info->targetValue || info->targetValue > batchEnd ||
        batchEnd >= (uint64_t(1) << 60)) return synInvalidArgument;
    auto* owner = synSingleton::getInstanceInternal();
    auto* source = dynamic_cast<ScalEvent*>(owner->getEventInterface(seed));
    auto* event = dynamic_cast<ScalEvent*>(owner->getEventInterface(destination));
    if (!source || !event) return synInvalidArgument;
    const ScalEvent sourceSnapshot(*source);
    if (sourceSnapshot.collectTime || !sourceSnapshot.isOnHclStream() || !sourceSnapshot.pStreamIfScal ||
        sourceSnapshot.hclSyncInfo.long_so_index != info->longSoIndex || sourceSnapshot.isInternalSignalingEvent())
        return synInvalidArgument;
    std::lock_guard<ScalEvent> lock(*event);
    if (event->collectTime || event->isInternalSignalingEvent() ||
        (event->pStreamIfScal && (!event->isOnHclStream() || event->pStreamIfScal != sourceSnapshot.pStreamIfScal)))
        return synInvalidArgument;
    event->clearState();
    event->pStreamIfScal = sourceSnapshot.pStreamIfScal;
    event->hclSyncInfo = sourceSnapshot.hclSyncInfo;
    event->hclSyncInfo.long_so_index = info->longSoIndex;
    event->hclSyncInfo.targetValue = info->targetValue;
    event->longSo = {info->longSoIndex, info->targetValue};
    event->setOnHclStream();
    event->setWaitMode(EventInterface::unset);
    return synSuccess;
}
