# SPDX-License-Identifier: Apache-2.0
"""Keep literal I32 operands resident instead of lowering I64 constant/cast pairs."""
import torch


def propagate_with_resident_buffers(module, inputs, propagate):
    """Fake propagation must convert real buffers introduced by this pass."""
    from torch._guards import detect_fake_mode

    mode = detect_fake_mode(inputs)
    if mode is None:
        return propagate()
    previous = mode.allow_non_fake_inputs
    try:
        mode.allow_non_fake_inputs = True
        return propagate()
    finally:
        mode.allow_non_fake_inputs = previous


_TENSOR_OPERATORS = {
    'add': torch.ops.aten.add.Tensor,
    'sub': torch.ops.aten.sub.Tensor,
    'mul': torch.ops.aten.mul.Tensor,
    'eq': torch.ops.aten.eq.Tensor,
    'ne': torch.ops.aten.ne.Tensor,
    'lt': torch.ops.aten.lt.Tensor,
    'le': torch.ops.aten.le.Tensor,
    'gt': torch.ops.aten.gt.Tensor,
    'ge': torch.ops.aten.ge.Tensor,
    'bitwise_and': torch.ops.aten.bitwise_and.Tensor,
    'bitwise_or': torch.ops.aten.bitwise_or.Tensor,
    'bitwise_xor': torch.ops.aten.bitwise_xor.Tensor,
    'bitwise_right_shift': torch.ops.aten.bitwise_right_shift.Tensor,
    'bitwise_left_shift': torch.ops.aten.bitwise_left_shift.Tensor,
    'remainder': torch.ops.aten.remainder.Tensor,
    'floor_divide': torch.ops.aten.floor_divide.default,
}


def retain_integer_constants(module, *, device_type='hpu'):
    """Only pure I32 tensor/literal expressions qualify; no I64 inputs narrow.

    The scalar allocation belongs to the compiled module for its entire lifetime.
    It has no mutable/request state and is shared by matching literals in this graph.
    """
    constants = {}
    changed = []
    for node in list(module.graph.nodes):
        if node.op != 'call_function' or len(node.args) < 2:
            continue
        schema = getattr(node.target, '_schema', None)
        if schema is None or schema.is_mutable:
            continue
        name = schema.name.removeprefix('aten::')
        if name not in _TENSOR_OPERATORS:
            continue
        source, literal = node.args[:2]
        if not isinstance(source, torch.fx.Node) or type(literal) is not int or not -(1 << 31) <= literal < (1 << 31):
            continue
        value = source.meta.get('val')
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int32 or value.device.type != device_type:
            continue
        # Shape/data pointers and user scalar tensors are never frozen.
        key = (literal, value.device)
        if key not in constants:
            from torch._subclasses.fake_tensor import unset_fake_temporarily
            with unset_fake_temporarily(), torch.inference_mode():
                scalar = torch.tensor(literal, dtype=torch.int32, device='cpu').to(value.device)
            attr = f'_decode_i32_constant_{len(constants)}'
            while hasattr(module, attr):
                attr += '_'
            module.register_buffer(attr, scalar, persistent=False)
            with module.graph.inserting_before(node):
                constant = module.graph.get_attr(attr)
            fake = value.new_empty((), dtype=torch.int32)
            # As in Bridge's normal placement pass, get_attr stays in the host
            # binding graph. The value is a persistent HPU input to the recipe;
            # embedding the buffer in a traced child would deepcopy HPU storage
            # while FakeTensorMode is active.
            constant.meta = dict(val=fake, placement='eager',
                                 output_device=value.device, output_dtypes=[torch.int32],
                                 output_layouts=[torch.strided], output_shapes=[[]], output_strides=[()],
                                 output_offset=[0], output_contiguous=[True])
            constants[key] = constant
        node.target = _TENSOR_OPERATORS[name]
        node.args = (source, constants[key], *node.args[2:])
        changed.append(dict(node=node.name, operation=name, literal=literal,
                            stack_trace=node.meta.get('stack_trace')))
    if changed:
        module.graph.lint()
        module.recompile()
    return dict(replaced_operands=len(changed), resident_constants=len(constants), nodes=changed)
