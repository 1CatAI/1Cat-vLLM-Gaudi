# SPDX-License-Identifier: Apache-2.0
"""Read actual compiled constant-section bytes in an initialized rank process."""
import ctypes


def inspect(path):
    sdk = ctypes.CDLL("libSynapse.so")
    handle = ctypes.c_void_p()
    sdk.synRecipeDeSerialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    sdk.synRecipeDeSerialize.restype = ctypes.c_int
    status = sdk.synRecipeDeSerialize(ctypes.byref(handle), str(path).encode())
    if status:
        raise RuntimeError(f"Recipe deserialize failed ({status}): {path}")
    try:
        # synapse_api_types.h: workspace=0, persistent tensors=1,
        # persistent bytes=4, compiled const-section bytes=5.
        attributes = (ctypes.c_int * 3)(1, 4, 5)
        values = (ctypes.c_uint64 * 3)()
        sdk.synRecipeGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                                             ctypes.POINTER(ctypes.c_int), ctypes.c_uint32,
                                             ctypes.c_void_p]
        sdk.synRecipeGetAttribute.restype = ctypes.c_int
        status = sdk.synRecipeGetAttribute(values, attributes, 3, handle)
        if status:
            raise RuntimeError(f"Constant-section attribute failed ({status}): {path}")
        return dict(path=str(path), persistent_tensors=values[0], persistent_bytes=values[1],
                    constant_section_bytes=values[2])
    finally:
        sdk.synRecipeDestroy.argtypes = [ctypes.c_void_p]
        sdk.synRecipeDestroy.restype = ctypes.c_int
        status = sdk.synRecipeDestroy(handle)
        if status:
            raise RuntimeError(f"Recipe cleanup failed ({status}): {path}")
