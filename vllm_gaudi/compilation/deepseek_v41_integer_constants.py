# SPDX-License-Identifier: Apache-2.0
"""Keep literal I32 operands resident instead of lowering I64 constant/cast pairs."""
import torch


def retain_static_factories(module, *, device_type='hpu', maximum_elements=4096):
    """Bind small, read-only coordinate/constant tensors at compilation time.

    Only factories with literal arguments qualify. Dynamic positions, RNG,
    uninitialized allocations and mutable factory results remain in the graph.
    This is an experimental pass; serving does not enable it by default.
    """
    factories = {torch.ops.aten.full.default, torch.ops.aten.zeros.default,
                 torch.ops.aten.ones.default, torch.ops.aten.arange.default,
                 torch.ops.aten.arange.start, torch.ops.aten.arange.start_step}
    changed = []

    def contains_node(value):
        if isinstance(value, torch.fx.Node):
            return True
        if isinstance(value, (tuple, list)):
            return any(contains_node(v) for v in value)
        if isinstance(value, dict):
            return any(contains_node(v) for v in value.values())
        return False

    def read_only(node, seen=None):
        seen = set() if seen is None else seen
        if node in seen:
            return True
        seen.add(node)
        for user in node.users:
            if user.op == 'output':
                return False
            schema = getattr(user.target, '_schema', None)
            if schema is None or schema.is_mutable:
                return False
            # Follow aliases too: mutating a view would mutate the constant.
            if any(v.alias_info is not None for v in schema.returns) and not read_only(user, seen):
                return False
        return True

    for node in list(module.graph.nodes):
        if node.op != 'call_function' or node.target not in factories:
            continue
        value = node.meta.get('val')
        if (not isinstance(value, torch.Tensor) or value.device.type != device_type
                or value.dtype not in (torch.int32, torch.int64, torch.float32)
                or any(type(d) is not int for d in value.shape)
                or value.numel() > maximum_elements
                or contains_node((node.args, node.kwargs)) or not read_only(node)):
            continue
        if (node.target in (torch.ops.aten.arange.default, torch.ops.aten.arange.start,
                            torch.ops.aten.arange.start_step) and value.dtype == torch.float32):
            # CPU and accelerator range construction may round differently.
            continue
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        kwargs = dict(node.kwargs, device='cpu')
        with unset_fake_temporarily(), torch.inference_mode():
            resident = node.target(*node.args, **kwargs).to(value.device)
        attr = f'_decode_static_factory_{len(changed)}'
        while hasattr(module, attr):
            attr += '_'
        module.register_buffer(attr, resident, persistent=False)
        with module.graph.inserting_before(node):
            constant = module.graph.get_attr(attr)
        constant.meta = dict(node.meta, placement='eager')
        node.replace_all_uses_with(constant)
        changed.append(dict(node=node.name, operation=str(node.target), elements=value.numel()))
        module.graph.erase_node(node)
    if changed:
        module.graph.lint()
        module.recompile()
    return dict(replaced_factories=len(changed), nodes=changed)


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


def retain_integer_clamp_bounds(module, *, device_type='hpu'):
    """Keep address bounds as I32 inputs to the stock gather/MME pipeline.

    Scalar clamp lowers through a complex node with internally materialized
    bounds. Tensor bounds preserve that clamp and its SRAM consumer while
    avoiding literal I64 construction. Only immutable in-range I32 bounds
    qualify; floating, dynamic and mutable clamps retain their original path.
    """
    changed, constants = [], {}
    variants = {torch.ops.aten.clamp.default: ('min', 'max'),
                torch.ops.aten.clamp_min.default: ('min',),
                torch.ops.aten.clamp_max.default: ('max',)}
    for node in list(module.graph.nodes):
        if node.op != 'call_function' or node.target not in variants or not node.args:
            continue
        source = node.args[0]
        value = source.meta.get('val') if isinstance(source, torch.fx.Node) else None
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int32 or value.device.type != device_type:
            continue
        names = variants[node.target]
        bounds = {name: node.args[index+1] if len(node.args)>index+1 else node.kwargs.get(name)
                  for index, name in enumerate(names)}
        if not any(v is not None for v in bounds.values()) or any(
                v is not None and (type(v) is not int or not -(1 << 31) <= v < (1 << 31)) for v in bounds.values()):
            continue
        operands = {'min': None, 'max': None}
        for name, literal in bounds.items():
            if literal is None:
                continue
            key = (literal, value.device)
            if key not in constants:
                from torch._subclasses.fake_tensor import unset_fake_temporarily
                with unset_fake_temporarily(), torch.inference_mode():
                    resident = torch.tensor(literal, dtype=torch.int32, device=value.device)
                attr = f'_decode_clamp_bound_{len(constants)}'
                while hasattr(module, attr):
                    attr += '_'
                module.register_buffer(attr, resident, persistent=False)
                with module.graph.inserting_before(node):
                    constant = module.graph.get_attr(attr)
                constant.meta = dict(val=value.new_empty(()), placement='eager',
                    output_device=value.device, output_dtypes=[torch.int32], output_layouts=[torch.strided],
                    output_shapes=[[]], output_strides=[()], output_offset=[0], output_contiguous=[True])
                constants[key] = constant
            operands[name] = constants[key]
        node.target = torch.ops.aten.clamp.Tensor
        node.args = (source, operands['min'], operands['max'])
        node.kwargs = {}
        changed.append(dict(node=node.name, bounds=bounds))
    if changed:
        module.graph.lint()
        module.recompile()
    return dict(replaced_clamps=len(changed), resident_bounds=len(constants), nodes=changed)
