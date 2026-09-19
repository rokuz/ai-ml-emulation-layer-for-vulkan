#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Generates synthetic TOSA graph modules (SPIR-V assembly, SPV_ARM_graph) that sweep the operators of the emulation
layer over ranks, element types, broadcast patterns and attribute values, and writes one manifest per suite for
mlel_opsuite."""
import argparse
import math
import os
import random
import sys
import zlib

ELEM = {
    'i8': ('OpTypeInt 8 0', 8, False, 'i8'),
    'i16': ('OpTypeInt 16 0', 16, False, 'i16'),
    'i32': ('OpTypeInt 32 0', 32, False, 'i32'),
    'i64': ('OpTypeInt 64 0', 64, False, None),
    'f16': ('OpTypeFloat 16', 16, True, 'f16'),
    'f32': ('OpTypeFloat 32', 32, True, 'f32'),
    'bool': ('OpTypeBool', 8, False, None),
}


def prod(dims):
    return math.prod(dims) if dims else 1


class Module:
    def __init__(self):
        self.decl = []
        self.cache = {}
        self.counter = 0
        self.caps = {'Shader', 'TensorsARM', 'GraphARM', 'VulkanMemoryModel'}
        self.ops = []

    def new(self, prefix='v'):
        self.counter += 1
        return f'%{prefix}{self.counter}'

    def get(self, key, make):
        if key in self.cache:
            return self.cache[key]
        ident = self.new()
        line = make()
        self.decl.append(f'{ident} = {line}')
        self.cache[key] = ident
        return ident

    def elem_t(self, e):
        if e == 'i8':
            self.caps.add('Int8')
        elif e == 'i16':
            self.caps.add('Int16')
        elif e == 'i64':
            self.caps.add('Int64')
        elif e == 'f16':
            self.caps.add('Float16')
        return self.get(('type', ELEM[e][0]), lambda: ELEM[e][0])

    def uint(self, v):
        t = self.elem_t('i32')
        return self.get(('const', 'i32', int(v) & 0xffffffff), lambda: f'OpConstant {t} {int(v) & 0xffffffff}')

    def boolc(self, v):
        t = self.elem_t('bool')
        return self.get(('bool', bool(v)), lambda: f'OpConstant{"True" if v else "False"} {t}')

    def scalar(self, e, v):
        if e == 'bool':
            return self.boolc(v)
        t = self.elem_t(e)
        bits = ELEM[e][1]
        if ELEM[e][2]:
            lit = repr(float(v))
        else:
            lit = str(int(v) & ((1 << bits) - 1))
        return self.get(('const', ELEM[e][0], lit), lambda: f'OpConstant {t} {lit}')

    def shape(self, dims):
        u = self.elem_t('i32')
        rank = self.uint(len(dims))
        arr = self.get(('array', len(dims)), lambda: f'OpTypeArray {u} {rank}')
        comps = [self.uint(d) for d in dims]
        return self.get(('shape', tuple(dims)), lambda: f'OpConstantComposite {arr} ' + ' '.join(comps))

    def tensor_t(self, e, dims):
        et = self.elem_t(e)
        rank = self.uint(len(dims))
        sh = self.shape(dims)
        return self.get(('tensor', e, tuple(dims)), lambda: f'OpTypeTensorARM {et} {rank} {sh}')

    def const(self, e, dims, values):
        """A constant tensor built from nested composites, `values` in row-major order."""
        values = list(values)
        assert len(values) == prod(dims), (dims, len(values))
        key = ('tconst', e, tuple(dims), tuple(values))
        if key in self.cache:
            return self.cache[key]
        t = self.tensor_t(e, dims)
        if len(dims) == 1:
            parts = [self.scalar(e, v) for v in values]
        else:
            inner = prod(dims[1:])
            parts = [self.const(e, dims[1:], values[i * inner:(i + 1) * inner]) for i in range(dims[0])]
        ident = self.new()
        self.decl.append(f'{ident} = OpConstantComposite {t} ' + ' '.join(parts))
        self.cache[key] = ident
        return ident

    def op(self, name, out_e, out_dims, *operands):
        t = self.tensor_t(out_e, out_dims)
        r = self.new('r')
        self.ops.append(f'{r} = OpExtInst {t} %tosa {name} ' + ' '.join(operands))
        return r

    def op2(self, name, out_e, out_dims, *operands):
        t = self.tensor_t(out_e, out_dims)
        st = self.new('st')
        self.decl.append(f'{st} = OpTypeStruct {t} {t}')
        r = self.new('r')
        self.ops.append(f'{r} = OpExtInst {st} %tosa {name} ' + ' '.join(operands))
        first, second = self.new('e'), self.new('e')
        self.ops.append(f'{first} = OpCompositeExtract {t} {r} 0')
        self.ops.append(f'{second} = OpCompositeExtract {t} {r} 1')
        return first, second


def render(m, inputs, input_ids, outputs):
    lines = [f'OpCapability {c}' for c in sorted(m.caps)]
    lines += ['OpExtension "SPV_ARM_tensors"', 'OpExtension "SPV_ARM_graph"', '%tosa = OpExtInstImport "TOSA.001000.1"',
              'OpMemoryModel Logical Vulkan']
    binding = 0
    in_vars = []
    out_vars = []
    for _ in inputs:
        v = f'%in_var{len(in_vars)}'
        lines += [f'OpDecorate {v} DescriptorSet 0', f'OpDecorate {v} Binding {binding}']
        in_vars.append(v)
        binding += 1
    for _ in outputs:
        v = f'%out_var{len(out_vars)}'
        lines += [f'OpDecorate {v} DescriptorSet 0', f'OpDecorate {v} Binding {binding}']
        out_vars.append(v)
        binding += 1
    in_types = [m.tensor_t(e, d) for e, d in inputs]
    out_types = [m.tensor_t(e, d) for _, e, d in outputs]
    indices = [m.uint(i) for i in range(max(len(inputs), len(outputs)))]
    body_tail = []
    lines += m.decl
    pointers = {}
    for v, t in list(zip(in_vars, in_types)) + list(zip(out_vars, out_types)):
        if t not in pointers:
            pointers[t] = f'%ptr{len(pointers)}'
            lines.append(f'{pointers[t]} = OpTypePointer UniformConstant {t}')
        lines.append(f'{v} = OpVariable {pointers[t]} UniformConstant')
    lines.append(f'%graph_type = OpTypeGraphARM {len(inputs)} ' + ' '.join(in_types + out_types))
    lines.append(f'OpGraphEntryPointARM %graph "main" ' + ' '.join(in_vars + out_vars))
    lines.append('%graph = OpGraphARM %graph_type')
    lines += body_tail
    for i, (ident, t) in enumerate(zip(input_ids, in_types)):
        lines.append(f'{ident} = OpGraphInputARM {t} {indices[i]}')
    lines += m.ops
    for j, (res, _, _) in enumerate(outputs):
        lines.append(f'OpGraphSetOutputARM {res} {indices[j]}')
    lines.append('OpGraphEndARM')
    return '\n'.join(lines) + '\n'


class Graph:
    def __init__(self, name):
        self.name = name
        self.m = Module()
        self.inputs = []
        self.input_ids = []
        self.outputs = []

    def input(self, e, dims, seed=None):
        ident = f'%input{len(self.inputs)}'
        self.inputs.append((e, list(dims), seed if seed is not None else (len(self.inputs) + 1) * 7919))
        self.input_ids.append(ident)
        return ident

    def output(self, res, e, dims):
        self.outputs.append((res, e, list(dims)))


GRAPHS = []
REFERENCES = {}


def add(g):
    GRAPHS.append(g)
    return g


def zp(g, e, value=0):
    return g.m.const(e, [1], [value])


UNARY_SHAPES = [[7], [3, 5], [2, 3, 5], [2, 3, 4, 5], [1, 2, 3, 2, 5], [2, 1, 3, 2, 2, 3], [300007], [3, 257, 131]]


def gen_unary():
    for dims in UNARY_SHAPES:
        tag = 'x'.join(map(str, dims))
        g = add(Graph(f'abs_i32_{tag}'))
        a = g.input('i32', dims)
        g.output(g.m.op('ABS', 'i32', dims, a), 'i32', dims)

        g = add(Graph(f'negate_i8_{tag}'))
        a = g.input('i8', dims)
        g.output(g.m.op('NEGATE', 'i8', dims, a, zp(g, 'i32'), zp(g, 'i32')), 'i8', dims)

        g = add(Graph(f'cast_i8_i32_{tag}'))
        a = g.input('i8', dims)
        g.output(g.m.op('CAST', 'i32', dims, a), 'i32', dims)

        g = add(Graph(f'sin_f32_{tag}'))
        a = g.input('f32', dims)
        g.output(g.m.op('SIN', 'f32', dims, a), 'f32', dims)

    for dims in [[3, 5], [2, 3, 4, 5]]:
        tag = 'x'.join(map(str, dims))
        for src, dst in [('i32', 'i8'), ('i16', 'f32'), ('f32', 'i16'), ('f16', 'f32'), ('f32', 'f16')]:
            g = add(Graph(f'cast_{src}_{dst}_{tag}'))
            a = g.input(src, dims)
            g.output(g.m.op('CAST', dst, dims, a), dst, dims)

        g = add(Graph(f'clamp_i8_{tag}'))
        a = g.input('i8', dims)
        g.output(g.m.op('CLAMP', 'i8', dims, g.m.scalar('i8', -20), g.m.scalar('i8', 37), g.m.uint(1), a), 'i8', dims)

        g = add(Graph(f'clamp_f32_{tag}'))
        a = g.input('f32', dims)
        g.output(g.m.op('CLAMP', 'f32', dims, g.m.scalar('f32', -1.5), g.m.scalar('f32', 2.25), g.m.uint(1), a), 'f32', dims)

        g = add(Graph(f'table_i8_{tag}'))
        a = g.input('i8', dims)
        table = [((i * 37) % 256) - 128 for i in range(256)]
        g.output(g.m.op('TABLE', 'i8', dims, a, g.m.const('i8', [256], table)), 'i8', dims)

        g = add(Graph(f'table_i16_{tag}'))
        a = g.input('i16', dims)
        table = [((i * 7919) % 65536) - 32768 for i in range(513)]
        g.output(g.m.op('TABLE', 'i32', dims, a, g.m.const('i16', [513], table)), 'i32', dims)


    for dims in [[3, 5], [2, 3, 4, 5], [1, 7, 6, 3]]:
        tag = 'x'.join(map(str, dims))
        for e in ['f32', 'f16']:
            g = add(Graph(f'negate_{e}_{tag}'))
            a = g.input(e, dims)
            g.output(g.m.op('NEGATE', e, dims, a, zp(g, e, 0.0), zp(g, e, 0.0)), e, dims)


BINARY_SHAPES = [
    ([7], [7]),
    ([3, 5], [3, 5]),
    ([3, 5], [1, 5]),
    ([3, 1], [1, 5]),
    ([2, 3, 4, 5], [2, 1, 4, 1]),
    ([1, 3, 1, 5], [2, 1, 4, 1]),
    ([2, 1, 3, 2, 5], [1, 4, 1, 2, 1]),
    ([2, 3, 1, 2, 3, 2], [1, 1, 4, 2, 1, 2]),
    ([300007], [1]),
    ([3, 257, 131], [3, 1, 131]),
]


def bshape(a, b):
    return [max(x, y) for x, y in zip(a, b)]


def gen_binary():
    for sa, sb in BINARY_SHAPES:
        out = bshape(sa, sb)
        tag = 'x'.join(map(str, sa)) + '_' + 'x'.join(map(str, sb))
        for opname in ['ADD', 'SUB']:
            g = add(Graph(f'{opname.lower()}_i32_{tag}'))
            a, b = g.input('i32', sa), g.input('i32', sb)
            g.output(g.m.op(opname, 'i32', out, a, b), 'i32', out)
        g = add(Graph(f'maximum_i8_{tag}'))
        a, b = g.input('i8', sa), g.input('i8', sb)
        g.output(g.m.op('MAXIMUM', 'i8', out, g.m.uint(1), a, b), 'i8', out)
        g = add(Graph(f'add_f32_{tag}'))
        a, b = g.input('f32', sa), g.input('f32', sb)
        g.output(g.m.op('ADD', 'f32', out, a, b), 'f32', out)
        g = add(Graph(f'mul_i32_shift_{tag}'))
        a, b = g.input('i16', sa), g.input('i16', sb)
        ca, cb = g.m.op('CAST', 'i32', sa, a), g.m.op('CAST', 'i32', sb, b)
        g.output(g.m.op('MUL', 'i32', out, ca, cb, g.m.const('i8', [1], [5])), 'i32', out)
        g = add(Graph(f'mul_i32_{tag}'))
        a, b = g.input('i32', sa), g.input('i32', sb)
        g.output(g.m.op('MUL', 'i32', out, a, b, g.m.const('i8', [1], [0])), 'i32', out)
        g = add(Graph(f'select_equal_i32_{tag}'))
        a, b = g.input('i8', sa), g.input('i8', sb)
        ca, cb = g.m.op('CAST', 'i32', sa, a), g.m.op('CAST', 'i32', sb, b)
        mask = g.m.op('GREATER_EQUAL', 'bool', out, ca, cb)
        g.output(g.m.op('SELECT', 'i32', out, mask, ca, cb), 'i32', out)
    for dims in [[3, 5], [2, 3, 4, 5]]:
        tag = 'x'.join(map(str, dims))
        for e in ['i8', 'i32']:
            for rnd in [False, True]:
                g = add(Graph(f'arshift_{e}_{"round" if rnd else "trunc"}_{tag}'))
                a = g.input(e, dims)
                bits = ELEM[e][1]
                shifts = g.m.const(e, dims, [(i * 7) % bits for i in range(prod(dims))])
                g.output(g.m.op('ARITHMETIC_RIGHT_SHIFT', e, dims, g.m.boolc(rnd), a, shifts), e, dims)

    for sa, sb in [([3, 5], [3, 5]), ([2, 3, 4, 7], [2, 3, 4, 7]), ([2, 3, 4, 5], [2, 1, 4, 1]), ([3, 6], [1, 1])]:
        out = bshape(sa, sb)
        tag = 'x'.join(map(str, sa)) + '_' + 'x'.join(map(str, sb))
        g = add(Graph(f'intdiv_i32_{tag}'))
        a, b = g.input('i8', sa), g.input('i8', sb)
        divisor = g.m.op('CLAMP', 'i8', sb, g.m.scalar('i8', 1), g.m.scalar('i8', 127), g.m.uint(1), b)
        ca, cb = g.m.op('CAST', 'i32', sa, a), g.m.op('CAST', 'i32', sb, divisor)
        g.output(g.m.op('INTDIV', 'i32', out, ca, cb), 'i32', out)


def gen_rescale():
    for dims in [[3, 5], [2, 3, 4, 5], [1, 17, 13, 12]]:
        tag = 'x'.join(map(str, dims))
        c = dims[-1]
        for src, dst, scale32, per_channel, rounding, izp, ozp in [
            ('i8', 'i8', True, False, 3, -128, 0),
            ('i8', 'i8', True, True, 1, 0, -128),
            ('i32', 'i8', True, True, 3, 0, 0),
            ('i16', 'i16', False, False, 1, 0, 0),
            ('i32', 'i16', False, True, 2, 0, 0),
            ('i8', 'i32', True, False, 1, 0, 0),
        ]:
            g = add(Graph(f'rescale_{src}_{dst}_{"s32" if scale32 else "s16"}_{"pc" if per_channel else "pt"}_r{rounding}_{tag}'))
            a = g.input(src, dims)
            n = c if per_channel else 1
            rng = random.Random(zlib.crc32(repr((src, dst, scale32, per_channel, tag)).encode()))
            if scale32:
                mult = g.m.const('i32', [n], [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(n)])
                shift = g.m.const('i8', [n], [rng.randrange(20, 40) for _ in range(n)])
            else:
                mult = g.m.const('i16', [n], [rng.randrange(1 << 13, (1 << 15) - 1) for _ in range(n)])
                shift = g.m.const('i8', [n], [rng.randrange(8, 24) for _ in range(n)])
            g.output(g.m.op('RESCALE', dst, dims, g.m.boolc(scale32), g.m.uint(rounding), g.m.boolc(per_channel), g.m.boolc(False),
                            g.m.boolc(False), a, mult, shift, zp(g, src, izp), zp(g, dst, ozp)), dst, dims)


def gen_layout():
    for dims, axis, count in [([5], 0, 2), ([3, 4], 1, 3), ([2, 3, 4, 5], 3, 2), ([2, 3, 4, 5], 0, 3), ([2, 3, 4, 5, 2], 2, 2)]:
        tag = 'x'.join(map(str, dims))
        g = add(Graph(f'concat_i8_ax{axis}_{count}_{tag}'))
        ins = []
        shapes = []
        for k in range(count):
            s = list(dims)
            s[axis] = dims[axis] + k
            shapes.append(s)
            ins.append(g.input('i8', s))
        out = list(dims)
        out[axis] = sum(s[axis] for s in shapes)
        g.output(g.m.op('CONCAT', 'i8', out, g.m.uint(axis), *ins), 'i8', out)

    for dims, start, size in [([9], [2], [5]), ([5, 7], [1, 2], [3, 4]), ([3, 6, 7, 5], [1, 2, 0, 1], [2, 3, 7, 3]),
                              ([2, 3, 4, 5, 6, 3], [1, 0, 1, 2, 3, 1], [1, 3, 2, 3, 2, 2])]:
        tag = 'x'.join(map(str, dims))
        g = add(Graph(f'slice_i16_{tag}'))
        a = g.input('i16', dims)
        g.output(g.m.op('SLICE', 'i16', size, a, g.m.const('i32', [len(dims)], start), g.m.const('i32', [len(dims)], size)), 'i16', size)

    for dims, padding in [([5], [2, 3]), ([3, 4], [1, 0, 2, 3]), ([1, 5, 6, 3], [0, 0, 1, 2, 3, 0, 0, 1])]:
        tag = 'x'.join(map(str, dims))
        out = [d + padding[2 * i] + padding[2 * i + 1] for i, d in enumerate(dims)]
        for e, value in [('i8', -7), ('f32', 0.5)]:
            g = add(Graph(f'pad_{e}_{tag}'))
            a = g.input(e, dims)
            g.output(g.m.op('PAD', e, out, a, g.m.const('i32', [len(padding)], padding), g.m.const(e, [1], [value])), e, out)

    for dims, shape in [([2, 3, 4], [4, 6]), ([6, 5], [1, 2, 3, 5]), ([2, 3, 2, 2, 5], [30, 4])]:
        tag = 'x'.join(map(str, dims))
        g = add(Graph(f'reshape_i32_{tag}_to_{"x".join(map(str, shape))}'))
        a = g.input('i32', dims)
        g.output(g.m.op('RESHAPE', 'i32', shape, a, g.m.const('i32', [len(shape)], shape)), 'i32', shape)

    for dims, perms in [([3, 5], [1, 0]), ([2, 3, 4], [2, 0, 1]), ([2, 3, 4, 5], [0, 3, 1, 2]), ([2, 3, 2, 4, 3], [4, 2, 0, 3, 1]),
                        ([2, 1, 3, 2, 2, 3], [5, 4, 3, 2, 1, 0])]:
        tag = 'x'.join(map(str, dims))
        out = [dims[p] for p in perms]
        g = add(Graph(f'transpose_i8_{tag}_{"".join(map(str, perms))}'))
        a = g.input('i8', dims)
        g.output(g.m.op('TRANSPOSE', 'i8', out, g.m.const('i32', [len(perms)], perms), a), 'i8', out)

    for dims, mult in [([5], [3]), ([2, 3], [2, 4]), ([1, 2, 3, 4], [2, 1, 3, 2])]:
        tag = 'x'.join(map(str, dims))
        out = [d * k for d, k in zip(dims, mult)]
        g = add(Graph(f'tile_i16_{tag}'))
        a = g.input('i16', dims)
        g.output(g.m.op('TILE', 'i16', out, a, g.m.const('i32', [len(mult)], mult)), 'i16', out)

    for dims in [[7], [3, 5], [2, 3, 4, 5], [2, 1, 3, 2, 2, 3]]:
        tag = 'x'.join(map(str, dims))
        for axis in sorted({0, len(dims) - 1, len(dims) // 2}):
            g = add(Graph(f'reverse_f32_ax{axis}_{tag}'))
            a = g.input('f32', dims)
            g.output(g.m.op('REVERSE', 'f32', dims, g.m.uint(axis), a), 'f32', dims)


def gen_spatial():
    for n, h, w, c in [(1, 9, 11, 3), (2, 16, 13, 5)]:
        for mode, scale, offset, border in [(1, [2, 1, 2, 1], [0, 0], [0, 0]), (2, [4, 2, 4, 2], [-1, -1], [1, 1]),
                                            (2, [3, 5, 5, 3], [0, 1], [0, 0]), (1, [1, 2, 1, 2], [0, 0], [0, 0])]:
            oh = ((h - 1) * scale[0] - offset[0] + border[0]) // scale[1] + 1
            ow = ((w - 1) * scale[2] - offset[1] + border[1]) // scale[3] + 1
            tag = f'{n}x{h}x{w}x{c}_m{mode}_s{"_".join(map(str, scale))}'
            out_e = 'i8' if mode == 1 else 'i32'
            g = add(Graph(f'resize_i8_{tag}'))
            a = g.input('i8', [n, h, w, c])
            g.output(g.m.op('RESIZE', out_e, [n, oh, ow, c], g.m.uint(mode), a, g.m.const('i32', [4], scale),
                            g.m.const('i32', [2], offset), g.m.const('i32', [2], border)), out_e, [n, oh, ow, c])

    for n, h, w, c in [(1, 9, 11, 3), (2, 16, 13, 6)]:
        for k, s, p in [([2, 2], [2, 2], [0, 0, 0, 0]), ([3, 3], [1, 1], [1, 1, 1, 1]), ([3, 2], [2, 1], [0, 1, 1, 0])]:
            oh = (h + p[0] + p[1] - k[0]) // s[0] + 1
            ow = (w + p[2] + p[3] - k[1]) // s[1] + 1
            tag = f'{n}x{h}x{w}x{c}_k{k[0]}{k[1]}_s{s[0]}{s[1]}_p{"".join(map(str, p))}'
            g = add(Graph(f'maxpool_i8_{tag}'))
            a = g.input('i8', [n, h, w, c])
            g.output(g.m.op('MAX_POOL2D', 'i8', [n, oh, ow, c], g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                            g.m.const('i32', [4], p), g.m.uint(1), a), 'i8', [n, oh, ow, c])
            g = add(Graph(f'avgpool_i8_{tag}'))
            a = g.input('i8', [n, h, w, c])
            g.output(g.m.op('AVG_POOL2D', 'i8', [n, oh, ow, c], g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                            g.m.const('i32', [4], p), g.m.uint(1), a, zp(g, 'i8', -3), zp(g, 'i8', 5)), 'i8', [n, oh, ow, c])

    for dims in [[7, 5], [2, 3, 4, 5], [2, 3, 2, 3, 4]]:
        tag = 'x'.join(map(str, dims))
        for axis in sorted({0, len(dims) - 1}):
            out = list(dims)
            out[axis] = 1
            g = add(Graph(f'reduce_sum_i32_ax{axis}_{tag}'))
            a = g.input('i8', dims)
            ca = g.m.op('CAST', 'i32', dims, a)
            g.output(g.m.op('REDUCE_SUM', 'i32', out, g.m.uint(axis), ca), 'i32', out)
            g = add(Graph(f'reduce_max_i8_ax{axis}_{tag}'))
            a = g.input('i8', dims)
            g.output(g.m.op('REDUCE_MAX', 'i8', out, g.m.uint(axis), g.m.uint(1), a), 'i8', out)
            if len(dims) > 1:
                out2 = [d for i, d in enumerate(dims) if i != axis]
                g = add(Graph(f'argmax_i8_ax{axis}_{tag}'))
                a = g.input('i8', dims)
                g.output(g.m.op('ARGMAX', 'i32', out2, g.m.uint(axis), g.m.uint(1), a), 'i32', out2)

    for n, k, c, w in [(1, 7, 3, 5), (2, 11, 4, 9)]:
        tag = f'{n}x{k}x{c}_w{w}'
        rng = random.Random(n * 1000 + k)
        g = add(Graph(f'gather_i16_{tag}'))
        a = g.input('i16', [n, k, c])
        idx = g.m.const('i32', [n, w], [rng.randrange(k) for _ in range(n * w)])
        g.output(g.m.op('GATHER', 'i16', [n, w, c], a, idx), 'i16', [n, w, c])
        g = add(Graph(f'scatter_i16_{tag}'))
        vin = g.input('i16', [n, k, c])
        upd = g.input('i16', [n, w if w <= k else k, c], seed=77)
        ww = min(w, k)
        idxs = []
        for _ in range(n):
            idxs += rng.sample(range(k), ww)
        g.output(g.m.op('SCATTER', 'i16', [n, k, c], vin, g.m.const('i32', [n, ww], idxs), upd), 'i16', [n, k, c])


def conv_out(size, pad_a, pad_b, k, stride, dilation):
    return (size - 1 + pad_a + pad_b - (k - 1) * dilation) // stride + 1


def gen_conv():
    rng = random.Random(1234)
    cases = [
        (1, 8, 9, 3, 5, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
        (1, 11, 10, 6, 8, 3, 3, [1, 0, 1, 0], [2, 2], [1, 1]),
        (2, 9, 9, 4, 7, 1, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
        (1, 12, 13, 5, 4, 5, 3, [2, 2, 1, 1], [1, 2], [1, 1]),
        (1, 13, 14, 2, 9, 3, 3, [2, 2, 2, 2], [1, 1], [2, 2]),
        (1, 7, 7, 16, 16, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
    ]
    for n, h, w, ic, oc, kh, kw, pad, stride, dil in cases:
        oh = conv_out(h, pad[0], pad[1], kh, stride[0], dil[0])
        ow = conv_out(w, pad[2], pad[3], kw, stride[1], dil[1])
        tag = f'{n}x{h}x{w}x{ic}_oc{oc}_k{kh}{kw}_s{stride[0]}{stride[1]}_d{dil[0]}_p{"".join(map(str, pad))}'
        for izp, wzp in [(-128, 0), (0, 0), (7, 0)]:
            g = add(Graph(f'conv2d_i8_z{izp}_{tag}'))
            a = g.input('i8', [n, h, w, ic])
            weights = g.m.const('i8', [oc, kh, kw, ic], [rng.randrange(-128, 128) for _ in range(oc * kh * kw * ic)])
            bias = g.m.const('i32', [oc], [rng.randrange(-5000, 5000) for _ in range(oc)])
            g.output(g.m.op('CONV2D', 'i32', [n, oh, ow, oc], g.m.const('i32', [4], pad), g.m.const('i32', [2], stride),
                            g.m.const('i32', [2], dil), g.m.uint(1), g.m.boolc(False), a, weights, bias, zp(g, 'i8', izp),
                            zp(g, 'i8', wzp)), 'i32', [n, oh, ow, oc])
        g = add(Graph(f'conv2d_f32_{tag}'))
        a = g.input('f32', [n, h, w, ic])
        weights = g.m.const('f32', [oc, kh, kw, ic], [rng.randrange(-16, 17) / 8 for _ in range(oc * kh * kw * ic)])
        bias = g.m.const('f32', [oc], [rng.randrange(-16, 17) / 4 for _ in range(oc)])
        g.output(g.m.op('CONV2D', 'f32', [n, oh, ow, oc], g.m.const('i32', [4], pad), g.m.const('i32', [2], stride),
                        g.m.const('i32', [2], dil), g.m.uint(3), g.m.boolc(False), a, weights, bias, zp(g, 'f32', 0.0),
                        zp(g, 'f32', 0.0)), 'f32', [n, oh, ow, oc])

    for n, h, w, c, m, kh, kw, pad, stride, dil in [(1, 9, 8, 3, 2, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
                                                     (1, 10, 11, 4, 1, 3, 3, [1, 0, 1, 0], [2, 2], [1, 1])]:
        oh = conv_out(h, pad[0], pad[1], kh, stride[0], dil[0])
        ow = conv_out(w, pad[2], pad[3], kw, stride[1], dil[1])
        tag = f'{n}x{h}x{w}x{c}_m{m}_k{kh}{kw}_s{stride[0]}{stride[1]}'
        g = add(Graph(f'dwconv2d_i8_{tag}'))
        a = g.input('i8', [n, h, w, c])
        weights = g.m.const('i8', [kh, kw, c, m], [rng.randrange(-128, 128) for _ in range(kh * kw * c * m)])
        bias = g.m.const('i32', [c * m], [rng.randrange(-5000, 5000) for _ in range(c * m)])
        g.output(g.m.op('DEPTHWISE_CONV2D', 'i32', [n, oh, ow, c * m], g.m.const('i32', [4], pad), g.m.const('i32', [2], stride),
                        g.m.const('i32', [2], dil), g.m.uint(1), g.m.boolc(False), a, weights, bias, zp(g, 'i8', -128),
                        zp(g, 'i8', 0)), 'i32', [n, oh, ow, c * m])

    for n, h, w, ic, oc, kh, kw, out_pad, stride in [(1, 5, 6, 3, 4, 3, 3, [0, 0, 0, 0], [2, 2]), (1, 4, 4, 2, 3, 4, 4, [-1, -1, -1, -1], [2, 2])]:
        oh = (h - 1) * stride[0] + out_pad[0] + out_pad[1] + kh
        ow = (w - 1) * stride[1] + out_pad[2] + out_pad[3] + kw
        tag = f'{n}x{h}x{w}x{ic}_oc{oc}_k{kh}{kw}_s{stride[0]}{stride[1]}_op{"".join(map(str, out_pad))}'
        g = add(Graph(f'tconv2d_i8_{tag}'))
        a = g.input('i8', [n, h, w, ic])
        weights = g.m.const('i8', [oc, kh, kw, ic], [rng.randrange(-128, 128) for _ in range(oc * kh * kw * ic)])
        bias = g.m.const('i32', [oc], [rng.randrange(-5000, 5000) for _ in range(oc)])
        g.output(g.m.op('TRANSPOSE_CONV2D', 'i32', [n, oh, ow, oc], g.m.const('i32', [4], out_pad), g.m.const('i32', [2], stride),
                        g.m.uint(1), g.m.boolc(False), a, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0)), 'i32', [n, oh, ow, oc])

    n, d, h, w, ic, oc, kd, kh, kw = 1, 5, 6, 7, 3, 4, 3, 3, 3
    pad, stride, dil = [1, 1, 1, 1, 1, 1], [1, 2, 1], [1, 1, 1]
    od, oh, ow = conv_out(d, 1, 1, kd, 1, 1), conv_out(h, 1, 1, kh, 2, 1), conv_out(w, 1, 1, kw, 1, 1)
    g = add(Graph(f'conv3d_i8_{n}x{d}x{h}x{w}x{ic}_oc{oc}'))
    a = g.input('i8', [n, d, h, w, ic])
    weights = g.m.const('i8', [oc, kd, kh, kw, ic], [rng.randrange(-128, 128) for _ in range(oc * kd * kh * kw * ic)])
    bias = g.m.const('i32', [oc], [rng.randrange(-5000, 5000) for _ in range(oc)])
    g.output(g.m.op('CONV3D', 'i32', [n, od, oh, ow, oc], g.m.const('i32', [6], pad), g.m.const('i32', [3], stride),
                    g.m.const('i32', [3], dil), g.m.uint(1), g.m.boolc(False), a, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0)),
             'i32', [n, od, oh, ow, oc])

    for n, h, c, w in [(1, 5, 7, 3), (2, 9, 13, 11), (3, 4, 16, 8)]:
        tag = f'{n}x{h}x{c}x{w}'
        g = add(Graph(f'matmul_i8_{tag}'))
        a, b = g.input('i8', [n, h, c]), g.input('i8', [n, c, w])
        g.output(g.m.op('MATMUL', 'i32', [n, h, w], a, b, zp(g, 'i8', -128), zp(g, 'i8', 0)), 'i32', [n, h, w])
        g = add(Graph(f'matmul_f32_{tag}'))
        a, b = g.input('f32', [n, h, c]), g.input('f32', [n, c, w])
        g.output(g.m.op('MATMUL', 'f32', [n, h, w], a, b, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0)), 'f32', [n, h, w])


def gen_bench():
    rng = random.Random(4321)
    big = [4, 512, 512, 16]
    tag = 'x'.join(map(str, big))

    g = add(Graph(f'bench_abs_i32_{tag}'))
    a = g.input('i32', big)
    g.output(g.m.op('ABS', 'i32', big, a), 'i32', big)

    g = add(Graph(f'bench_cast_i8_i32_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('CAST', 'i32', big, a), 'i32', big)

    g = add(Graph(f'bench_clamp_f32_{tag}'))
    a = g.input('f32', big)
    g.output(g.m.op('CLAMP', 'f32', big, g.m.scalar('f32', -1.5), g.m.scalar('f32', 2.25), g.m.uint(1), a), 'f32', big)

    g = add(Graph(f'bench_abs_f16_{tag}'))
    a = g.input('f16', big)
    g.output(g.m.op('ABS', 'f16', big, a), 'f16', big)

    g = add(Graph(f'bench_add_f16_same_{tag}'))
    a, b = g.input('f16', big), g.input('f16', big)
    g.output(g.m.op('ADD', 'f16', big, a, b), 'f16', big)

    g = add(Graph(f'bench_bitwise_not_i8_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('BITWISE_NOT', 'i8', big, a), 'i8', big)

    g = add(Graph(f'bench_maximum_i8_same_{tag}'))
    a, b = g.input('i8', big), g.input('i8', big)
    g.output(g.m.op('MAXIMUM', 'i8', big, g.m.uint(1), a, b), 'i8', big)

    g = add(Graph(f'bench_table_i8_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('TABLE', 'i8', big, a, g.m.const('i8', [256], [((i * 37) % 256) - 128 for i in range(256)])), 'i8', big)

    g = add(Graph(f'bench_add_i32_same_{tag}'))
    a, b = g.input('i32', big), g.input('i32', big)
    g.output(g.m.op('ADD', 'i32', big, a, b), 'i32', big)

    g = add(Graph(f'bench_add_f32_broadcast_{tag}'))
    a, b = g.input('f32', big), g.input('f32', [1, 512, 1, 16])
    g.output(g.m.op('ADD', 'f32', big, a, b), 'f32', big)

    g = add(Graph(f'bench_mul_i32_shift_{tag}'))
    a, b = g.input('i32', big), g.input('i32', [1, 1, 1, 16])
    g.output(g.m.op('MUL', 'i32', big, a, b, g.m.const('i8', [1], [7])), 'i32', big)

    g = add(Graph(f'bench_rescale_i32_i8_pc_{tag}'))
    a = g.input('i32', big)
    mult = g.m.const('i32', [16], [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(16)])
    shift = g.m.const('i8', [16], [rng.randrange(34, 40) for _ in range(16)])
    g.output(g.m.op('RESCALE', 'i8', big, g.m.boolc(True), g.m.uint(3), g.m.boolc(True), g.m.boolc(False), g.m.boolc(False), a,
                    mult, shift, zp(g, 'i32', 0), zp(g, 'i8', -128)), 'i8', big)

    g = add(Graph(f'bench_concat_i8_ax3_{tag}'))
    a, b = g.input('i8', [4, 512, 512, 8]), g.input('i8', [4, 512, 512, 8])
    g.output(g.m.op('CONCAT', 'i8', big, g.m.uint(3), a, b), 'i8', big)

    g = add(Graph(f'bench_slice_i8_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('SLICE', 'i8', [4, 500, 500, 16], a, g.m.const('i32', [4], [0, 6, 6, 0]),
                    g.m.const('i32', [4], [4, 500, 500, 16])), 'i8', [4, 500, 500, 16])

    g = add(Graph(f'bench_pad_i8_{tag}'))
    a = g.input('i8', [4, 500, 500, 16])
    g.output(g.m.op('PAD', 'i8', big, a, g.m.const('i32', [8], [0, 0, 6, 6, 6, 6, 0, 0]), g.m.const('i8', [1], [-3])), 'i8', big)

    g = add(Graph(f'bench_transpose_i8_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('TRANSPOSE', 'i8', [4, 16, 512, 512], g.m.const('i32', [4], [0, 3, 1, 2]), a), 'i8', [4, 16, 512, 512])

    g = add(Graph(f'bench_reverse_i8_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('REVERSE', 'i8', big, g.m.uint(1), a), 'i8', big)

    g = add(Graph('bench_tile_i8_4x256x256x8'))
    a = g.input('i8', [4, 256, 256, 8])
    g.output(g.m.op('TILE', 'i8', big, a, g.m.const('i32', [4], [1, 2, 2, 2])), 'i8', big)

    for mode, out_e in [(1, 'i8'), (2, 'i32')]:
        g = add(Graph(f'bench_resize_i8_m{mode}_4x256x256x16'))
        a = g.input('i8', [4, 256, 256, 16])
        g.output(g.m.op('RESIZE', out_e, [4, 511, 511, 16], g.m.uint(mode), a, g.m.const('i32', [4], [2, 1, 2, 1]),
                        g.m.const('i32', [2], [0, 0]), g.m.const('i32', [2], [0, 0])), out_e, [4, 511, 511, 16])

    g = add(Graph(f'bench_maxpool_i8_k3_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('MAX_POOL2D', 'i8', big, g.m.const('i32', [2], [3, 3]), g.m.const('i32', [2], [1, 1]),
                    g.m.const('i32', [4], [1, 1, 1, 1]), g.m.uint(1), a), 'i8', big)

    g = add(Graph(f'bench_avgpool_i8_k3_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('AVG_POOL2D', 'i8', big, g.m.const('i32', [2], [3, 3]), g.m.const('i32', [2], [1, 1]),
                    g.m.const('i32', [4], [1, 1, 1, 1]), g.m.uint(1), a, zp(g, 'i8', 0), zp(g, 'i8', 0)), 'i8', big)

    g = add(Graph(f'bench_reduce_sum_i32_ax3_{tag}'))
    a = g.input('i32', big)
    g.output(g.m.op('REDUCE_SUM', 'i32', [4, 512, 512, 1], g.m.uint(3), a), 'i32', [4, 512, 512, 1])

    g = add(Graph(f'bench_argmax_i8_ax3_{tag}'))
    a = g.input('i8', big)
    g.output(g.m.op('ARGMAX', 'i32', [4, 512, 512], g.m.uint(3), g.m.uint(1), a), 'i32', [4, 512, 512])

    for e in ['i8', 'f32']:
        g = add(Graph(f'bench_matmul_{e}_8x256x256x256'))
        a, b = g.input(e, [8, 256, 256]), g.input(e, [8, 256, 256])
        out_e = 'i32' if e == 'i8' else 'f32'
        zv = 0.0 if e == 'f32' else 0
        g.output(g.m.op('MATMUL', out_e, [8, 256, 256], a, b, zp(g, e, zv), zp(g, e, zv)), out_e, [8, 256, 256])

    ic, oc = 32, 32
    for e in ['i8', 'f32']:
        g = add(Graph(f'bench_conv2d_{e}_1x256x256x32_oc32_k3'))
        a = g.input(e, [1, 256, 256, ic])
        if e == 'i8':
            weights = g.m.const('i8', [oc, 3, 3, ic], [rng.randrange(-128, 128) for _ in range(oc * 9 * ic)])
            bias = g.m.const('i32', [oc], [rng.randrange(-5000, 5000) for _ in range(oc)])
            out_e, acc, izp, wzp = 'i32', 1, zp(g, 'i8', 0), zp(g, 'i8', 0)
        else:
            weights = g.m.const('f32', [oc, 3, 3, ic], [rng.randrange(-16, 17) / 8 for _ in range(oc * 9 * ic)])
            bias = g.m.const('f32', [oc], [rng.randrange(-16, 17) / 4 for _ in range(oc)])
            out_e, acc, izp, wzp = 'f32', 3, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0)
        g.output(g.m.op('CONV2D', out_e, [1, 256, 256, oc], g.m.const('i32', [4], [1, 1, 1, 1]), g.m.const('i32', [2], [1, 1]),
                        g.m.const('i32', [2], [1, 1]), g.m.uint(acc), g.m.boolc(False), a, weights, bias, izp, wzp),
                 out_e, [1, 256, 256, oc])

    g = add(Graph('bench_dwconv2d_i8_1x512x512x32_k3'))
    a = g.input('i8', [1, 512, 512, 32])
    weights = g.m.const('i8', [3, 3, 32, 1], [rng.randrange(-128, 128) for _ in range(9 * 32)])
    bias = g.m.const('i32', [32], [rng.randrange(-5000, 5000) for _ in range(32)])
    g.output(g.m.op('DEPTHWISE_CONV2D', 'i32', [1, 512, 512, 32], g.m.const('i32', [4], [1, 1, 1, 1]), g.m.const('i32', [2], [1, 1]),
                    g.m.const('i32', [2], [1, 1]), g.m.uint(1), g.m.boolc(False), a, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0)),
             'i32', [1, 512, 512, 32])

    g = add(Graph('bench_tconv2d_i8_1x128x128x16_oc16_s2'))
    a = g.input('i8', [1, 128, 128, 16])
    weights = g.m.const('i8', [16, 4, 4, 16], [rng.randrange(-128, 128) for _ in range(16 * 16 * 16)])
    bias = g.m.const('i32', [16], [rng.randrange(-5000, 5000) for _ in range(16)])
    g.output(g.m.op('TRANSPOSE_CONV2D', 'i32', [1, 258, 258, 16], g.m.const('i32', [4], [0, 0, 0, 0]), g.m.const('i32', [2], [2, 2]),
                    g.m.uint(1), g.m.boolc(False), a, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0)), 'i32', [1, 258, 258, 16])

    g = add(Graph('bench_model_cnn_block_i8'))
    x = g.input('i8', [1, 256, 256, 16])

    def conv_q(inp, h, w, c_in, c_out):
        wts = g.m.const('i8', [c_out, 3, 3, c_in], [rng.randrange(-128, 128) for _ in range(c_out * 9 * c_in)])
        bs = g.m.const('i32', [c_out], [rng.randrange(-5000, 5000) for _ in range(c_out)])
        acc_out = g.m.op('CONV2D', 'i32', [1, h, w, c_out], g.m.const('i32', [4], [1, 1, 1, 1]), g.m.const('i32', [2], [1, 1]),
                         g.m.const('i32', [2], [1, 1]), g.m.uint(1), g.m.boolc(False), inp, wts, bs, zp(g, 'i8', -128), zp(g, 'i8', 0))
        mult = g.m.const('i32', [c_out], [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(c_out)])
        shift = g.m.const('i8', [c_out], [rng.randrange(40, 44) for _ in range(c_out)])
        q = g.m.op('RESCALE', 'i8', [1, h, w, c_out], g.m.boolc(True), g.m.uint(3), g.m.boolc(True), g.m.boolc(False), g.m.boolc(False),
                   acc_out, mult, shift, zp(g, 'i32', 0), zp(g, 'i8', -128))
        return g.m.op('CLAMP', 'i8', [1, h, w, c_out], g.m.scalar('i8', -128), g.m.scalar('i8', 127), g.m.uint(1), q)

    c1 = conv_q(x, 256, 256, 16, 32)
    p1 = g.m.op('MAX_POOL2D', 'i8', [1, 128, 128, 32], g.m.const('i32', [2], [2, 2]), g.m.const('i32', [2], [2, 2]),
                g.m.const('i32', [4], [0, 0, 0, 0]), g.m.uint(1), c1)
    c2 = conv_q(p1, 128, 128, 32, 32)
    up = g.m.op('RESIZE', 'i8', [1, 256, 256, 32], g.m.uint(1), c2, g.m.const('i32', [4], [2, 1, 2, 1]), g.m.const('i32', [2], [0, 0]),
                g.m.const('i32', [2], [1, 1]))
    cat = g.m.op('CONCAT', 'i8', [1, 256, 256, 64], g.m.uint(3), c1, up)
    g.output(cat, 'i8', [1, 256, 256, 64])

    g = add(Graph('bench_model_attention_f32'))
    n, t, d = 8, 256, 64
    x = g.input('f32', [n, t, d])
    wq = g.m.const('f32', [n, d, d], [rng.randrange(-8, 9) / 64 for _ in range(n * d * d)])
    wk = g.m.const('f32', [n, d, d], [rng.randrange(-8, 9) / 64 for _ in range(n * d * d)])
    q = g.m.op('MATMUL', 'f32', [n, t, d], x, wq, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0))
    k = g.m.op('MATMUL', 'f32', [n, t, d], x, wk, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0))
    kt = g.m.op('TRANSPOSE', 'f32', [n, d, t], g.m.const('i32', [3], [0, 2, 1]), k)
    att = g.m.op('MATMUL', 'f32', [n, t, t], q, kt, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0))
    att = g.m.op('CLAMP', 'f32', [n, t, t], g.m.scalar('f32', -8.0), g.m.scalar('f32', 8.0), g.m.uint(1), att)
    ex = g.m.op('EXP', 'f32', [n, t, t], att)
    sm = g.m.op('REDUCE_SUM', 'f32', [n, t, 1], g.m.uint(2), ex)
    inv = g.m.op('RECIPROCAL', 'f32', [n, t, 1], sm)
    probs = g.m.op('MUL', 'f32', [n, t, t], ex, inv, g.m.const('i8', [1], [0]))
    ctx_out = g.m.op('MATMUL', 'f32', [n, t, d], probs, x, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0))
    g.output(g.m.op('ADD', 'f32', [n, t, d], ctx_out, x), 'f32', [n, t, d])

    g = add(Graph('bench_model_elementwise_i8'))
    x = g.input('i8', big)
    c = g.m.op('CAST', 'i32', big, x)
    scale = g.m.const('i32', [1, 1, 1, 16], [rng.randrange(1, 300) for _ in range(16)])
    m = g.m.op('MUL', 'i32', big, c, scale, g.m.const('i8', [1], [4]))
    b = g.m.op('ADD', 'i32', big, m, g.m.const('i32', [1, 1, 1, 16], [rng.randrange(-500, 500) for _ in range(16)]))
    mult = g.m.const('i32', [16], [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(16)])
    shift = g.m.const('i8', [16], [rng.randrange(30, 34) for _ in range(16)])
    r = g.m.op('RESCALE', 'i8', big, g.m.boolc(True), g.m.uint(3), g.m.boolc(True), g.m.boolc(False), g.m.boolc(False), b, mult,
               shift, zp(g, 'i32', 0), zp(g, 'i8', 0))
    tb = g.m.op('TABLE', 'i8', big, r, g.m.const('i8', [256], [((i * 91) % 256) - 128 for i in range(256)]))
    g.output(g.m.op('CLAMP', 'i8', big, g.m.scalar('i8', -100), g.m.scalar('i8', 100), g.m.uint(1), tb), 'i8', big)

RESCALE_VARIANTS = [
    (True, True, 3, 0, -128, 'i8'),
    (False, True, 1, 0, 0, 'i8'),
    (True, False, 2, 0, 3, 'i16'),
    (True, True, 1, 5, -7, 'i32'),
    (False, False, 3, 0, 100, 'i8'),
]


def rescale_after(g, rng, x, dims, variant):
    per_channel, scale32, rounding, izp, ozp, dst = variant
    n = dims[-1] if per_channel else 1
    if scale32:
        mult = g.m.const('i32', [n], [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(n)])
        shift = g.m.const('i8', [n], [rng.randrange(36, 46) for _ in range(n)])
    else:
        mult = g.m.const('i16', [n], [rng.randrange(1 << 13, (1 << 15) - 1) for _ in range(n)])
        shift = g.m.const('i8', [n], [rng.randrange(20, 30) for _ in range(n)])
    return g.m.op('RESCALE', dst, dims, g.m.boolc(scale32), g.m.uint(rounding), g.m.boolc(per_channel), g.m.boolc(False),
                  g.m.boolc(False), x, mult, shift, zp(g, 'i32', izp), zp(g, dst, ozp))


def conv2d_i8(g, rng, x, n, h, w, ic, oc, k, pad, stride, dil, izp=-128, wzp=0):
    oh = conv_out(h, pad[0], pad[1], k, stride[0], dil[0])
    ow = conv_out(w, pad[2], pad[3], k, stride[1], dil[1])
    weights = g.m.const('i8', [oc, k, k, ic], [rng.randrange(-128, 128) for _ in range(oc * k * k * ic)])
    bias = g.m.const('i32', [oc], [rng.randrange(-50000, 50000) for _ in range(oc)])
    dims = [n, oh, ow, oc]
    r = g.m.op('CONV2D', 'i32', dims, g.m.const('i32', [4], pad), g.m.const('i32', [2], stride), g.m.const('i32', [2], dil),
               g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', izp), zp(g, 'i8', wzp))
    return r, dims


def gen_fusion():
    rng = random.Random(2024)
    for n, h, w, ic, oc, k, pad, stride, dil in [(1, 9, 10, 3, 5, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
                                                (2, 11, 9, 8, 8, 3, [0, 1, 1, 0], [2, 2], [1, 1]),
                                                (1, 7, 8, 4, 7, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
                                                (1, 12, 11, 5, 16, 3, [2, 2, 2, 2], [1, 1], [2, 2])]:
        for vi, variant in enumerate(RESCALE_VARIANTS):
            g = add(Graph(f'fusion_conv2d_{n}x{h}x{w}x{ic}_oc{oc}_k{k}_s{stride[0]}_d{dil[0]}_v{vi}'))
            x = g.input('i8', [n, h, w, ic])
            r, dims = conv2d_i8(g, rng, x, n, h, w, ic, oc, k, pad, stride, dil, izp=[-128, 0, 7][vi % 3])
            g.output(rescale_after(g, rng, r, dims, variant), variant[5], dims)

    for n, h, w, c, m, stride in [(1, 9, 8, 3, 2, [1, 1]), (2, 10, 11, 4, 1, [2, 2])]:
        for vi in (0, 2, 3):
            variant = RESCALE_VARIANTS[vi]
            g = add(Graph(f'fusion_dwconv2d_{n}x{h}x{w}x{c}_m{m}_s{stride[0]}_v{vi}'))
            x = g.input('i8', [n, h, w, c])
            oh, ow = conv_out(h, 1, 1, 3, stride[0], 1), conv_out(w, 1, 1, 3, stride[1], 1)
            weights = g.m.const('i8', [3, 3, c, m], [rng.randrange(-128, 128) for _ in range(9 * c * m)])
            bias = g.m.const('i32', [c * m], [rng.randrange(-50000, 50000) for _ in range(c * m)])
            dims = [n, oh, ow, c * m]
            r = g.m.op('DEPTHWISE_CONV2D', 'i32', dims, g.m.const('i32', [4], [1, 1, 1, 1]), g.m.const('i32', [2], stride),
                       g.m.const('i32', [2], [1, 1]), g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', -128),
                       zp(g, 'i8', 0))
            g.output(rescale_after(g, rng, r, dims, variant), variant[5], dims)

    for n, h, w, ic, oc, kk, out_pad in [(1, 5, 6, 3, 4, 3, [0, 0, 0, 0]), (1, 4, 4, 2, 6, 4, [-1, -1, -1, -1])]:
        for vi in (0, 4):
            variant = RESCALE_VARIANTS[vi]
            g = add(Graph(f'fusion_tconv2d_{n}x{h}x{w}x{ic}_oc{oc}_k{kk}_v{vi}'))
            x = g.input('i8', [n, h, w, ic])
            oh = (h - 1) * 2 + out_pad[0] + out_pad[1] + kk
            ow = (w - 1) * 2 + out_pad[2] + out_pad[3] + kk
            weights = g.m.const('i8', [oc, kk, kk, ic], [rng.randrange(-128, 128) for _ in range(oc * kk * kk * ic)])
            bias = g.m.const('i32', [oc], [rng.randrange(-50000, 50000) for _ in range(oc)])
            dims = [n, oh, ow, oc]
            r = g.m.op('TRANSPOSE_CONV2D', 'i32', dims, g.m.const('i32', [4], out_pad), g.m.const('i32', [2], [2, 2]),
                       g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0))
            g.output(rescale_after(g, rng, r, dims, variant), variant[5], dims)

    for vi in (0, 3):
        variant = RESCALE_VARIANTS[vi]
        g = add(Graph(f'fusion_conv3d_1x5x6x7x3_oc4_v{vi}'))
        x = g.input('i8', [1, 5, 6, 7, 3])
        od, oh, ow = conv_out(5, 1, 1, 3, 1, 1), conv_out(6, 1, 1, 3, 2, 1), conv_out(7, 1, 1, 3, 1, 1)
        weights = g.m.const('i8', [4, 3, 3, 3, 3], [rng.randrange(-128, 128) for _ in range(4 * 27 * 3)])
        bias = g.m.const('i32', [4], [rng.randrange(-50000, 50000) for _ in range(4)])
        dims = [1, od, oh, ow, 4]
        r = g.m.op('CONV3D', 'i32', dims, g.m.const('i32', [6], [1, 1, 1, 1, 1, 1]), g.m.const('i32', [3], [1, 2, 1]),
                   g.m.const('i32', [3], [1, 1, 1]), g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', 0),
                   zp(g, 'i8', 0))
        g.output(rescale_after(g, rng, r, dims, variant), variant[5], dims)

    for n, h, c, w in [(1, 5, 7, 3), (3, 4, 16, 9)]:
        for vi in (0, 1, 2):
            variant = RESCALE_VARIANTS[vi]
            g = add(Graph(f'fusion_matmul_{n}x{h}x{c}x{w}_v{vi}'))
            a, b = g.input('i8', [n, h, c]), g.input('i8', [n, c, w])
            r = g.m.op('MATMUL', 'i32', [n, h, w], a, b, zp(g, 'i8', -128), zp(g, 'i8', 3))
            g.output(rescale_after(g, rng, r, [n, h, w], variant), variant[5], [n, h, w])

    g = add(Graph('fusion_negative_conv2d_is_output'))
    x = g.input('i8', [1, 9, 10, 3])
    r, dims = conv2d_i8(g, rng, x, 1, 9, 10, 3, 5, 3, [1, 1, 1, 1], [1, 1], [1, 1])
    g.output(r, 'i32', dims)
    g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)

    g = add(Graph('fusion_negative_two_rescales'))
    x = g.input('i8', [1, 9, 10, 3])
    r, dims = conv2d_i8(g, rng, x, 1, 9, 10, 3, 5, 3, [1, 1, 1, 1], [1, 1], [1, 1])
    g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)
    g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[2]), 'i16', dims)

    g = add(Graph('fusion_negative_unsigned_output'))
    x = g.input('i8', [1, 9, 10, 3])
    r, dims = conv2d_i8(g, rng, x, 1, 9, 10, 3, 5, 3, [1, 1, 1, 1], [1, 1], [1, 1])
    mult = g.m.const('i32', [5], [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(5)])
    shift = g.m.const('i8', [5], [rng.randrange(36, 46) for _ in range(5)])
    g.output(g.m.op('RESCALE', 'i8', dims, g.m.boolc(True), g.m.uint(3), g.m.boolc(True), g.m.boolc(False), g.m.boolc(True),
                    r, mult, shift, zp(g, 'i32', 0), zp(g, 'i8', 0)), 'i8', dims)

    g = add(Graph('fusion_chain_conv_rescale_conv_rescale_clamp'))
    x = g.input('i8', [2, 16, 15, 8])
    r, dims = conv2d_i8(g, rng, x, 2, 16, 15, 8, 8, 3, [1, 1, 1, 1], [1, 1], [1, 1])
    q = rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0])
    r2, dims2 = conv2d_i8(g, rng, q, 2, 16, 15, 8, 12, 3, [1, 1, 1, 1], [2, 2], [1, 1])
    q2 = rescale_after(g, rng, r2, dims2, RESCALE_VARIANTS[1])
    g.output(g.m.op('CLAMP', 'i8', dims2, g.m.scalar('i8', -100), g.m.scalar('i8', 90), g.m.uint(1), q2), 'i8', dims2)


def gen_fusion_bench():
    rng = random.Random(77)

    g = add(Graph('bench_fusion_conv2d_i8_rescale_1x256x256x32_oc32'))
    x = g.input('i8', [1, 256, 256, 32])
    r, dims = conv2d_i8(g, rng, x, 1, 256, 256, 32, 32, 3, [1, 1, 1, 1], [1, 1], [1, 1])
    g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)

    g = add(Graph('bench_fusion_dwconv2d_i8_rescale_1x512x512x32'))
    x = g.input('i8', [1, 512, 512, 32])
    weights = g.m.const('i8', [3, 3, 32, 1], [rng.randrange(-128, 128) for _ in range(9 * 32)])
    bias = g.m.const('i32', [32], [rng.randrange(-50000, 50000) for _ in range(32)])
    dims = [1, 512, 512, 32]
    r = g.m.op('DEPTHWISE_CONV2D', 'i32', dims, g.m.const('i32', [4], [1, 1, 1, 1]), g.m.const('i32', [2], [1, 1]),
               g.m.const('i32', [2], [1, 1]), g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', -128),
               zp(g, 'i8', 0))
    g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)

    g = add(Graph('bench_fusion_tconv2d_i8_rescale_1x128x128x16_oc16'))
    x = g.input('i8', [1, 128, 128, 16])
    weights = g.m.const('i8', [16, 4, 4, 16], [rng.randrange(-128, 128) for _ in range(16 * 16 * 16)])
    bias = g.m.const('i32', [16], [rng.randrange(-50000, 50000) for _ in range(16)])
    dims = [1, 258, 258, 16]
    r = g.m.op('TRANSPOSE_CONV2D', 'i32', dims, g.m.const('i32', [4], [0, 0, 0, 0]), g.m.const('i32', [2], [2, 2]),
               g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0))
    g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)

    g = add(Graph('bench_fusion_matmul_i8_rescale_8x256x256x256'))
    a, b = g.input('i8', [8, 256, 256]), g.input('i8', [8, 256, 256])
    r = g.m.op('MATMUL', 'i32', [8, 256, 256], a, b, zp(g, 'i8', -128), zp(g, 'i8', 0))
    g.output(rescale_after(g, rng, r, [8, 256, 256], RESCALE_VARIANTS[0]), 'i8', [8, 256, 256])


def gen_conv_wide():
    rng = random.Random(1616)
    for n, h, w, ic, oc, k, pad, stride, dil in [(1, 9, 10, 16, 5, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
                                                (1, 9, 10, 17, 32, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
                                                (2, 11, 9, 31, 7, 3, [0, 1, 1, 0], [2, 2], [1, 1]),
                                                (1, 8, 12, 32, 32, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
                                                (1, 12, 11, 35, 6, 3, [2, 2, 2, 2], [1, 1], [2, 2]),
                                                (1, 7, 9, 48, 16, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
                                                (2, 10, 10, 64, 9, 3, [1, 0, 1, 0], [2, 2], [1, 1])]:
        tag = f'{n}x{h}x{w}x{ic}_oc{oc}_k{k}_s{stride[0]}_d{dil[0]}'
        for izp in (-128, 0, 7):
            g = add(Graph(f'conv_wide_i8_z{izp}_{tag}'))
            x = g.input('i8', [n, h, w, ic])
            r, dims = conv2d_i8(g, rng, x, n, h, w, ic, oc, k, pad, stride, dil, izp=izp)
            g.output(r, 'i32', dims)
        g = add(Graph(f'conv_wide_i8_rescale_{tag}'))
        x = g.input('i8', [n, h, w, ic])
        r, dims = conv2d_i8(g, rng, x, n, h, w, ic, oc, k, pad, stride, dil)
        g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)

        g = add(Graph(f'conv_wide_f32_{tag}'))
        x = g.input('f32', [n, h, w, ic])
        oh = conv_out(h, pad[0], pad[1], k, stride[0], dil[0])
        ow = conv_out(w, pad[2], pad[3], k, stride[1], dil[1])
        weights = g.m.const('f32', [oc, k, k, ic], [rng.randrange(-16, 17) / 8 for _ in range(oc * k * k * ic)])
        bias = g.m.const('f32', [oc], [rng.randrange(-16, 17) / 4 for _ in range(oc)])
        g.output(g.m.op('CONV2D', 'f32', [n, oh, ow, oc], g.m.const('i32', [4], pad), g.m.const('i32', [2], stride),
                        g.m.const('i32', [2], dil), g.m.uint(3), g.m.boolc(False), x, weights, bias, zp(g, 'f32', 0.0),
                        zp(g, 'f32', 0.0)), 'f32', [n, oh, ow, oc])


def gen_conv_shapes():
    rng = random.Random(4096)
    for n, h, w, ic, oc, k, stride in [(1, 544, 960, 16, 4, 3, 1),
                                       (1, 272, 480, 12, 32, 3, 2),
                                       (1, 272, 480, 32, 32, 3, 1),
                                       (1, 272, 480, 48, 16, 3, 1),
                                       (1, 136, 240, 32, 32, 3, 2),
                                       (1, 136, 240, 32, 36, 3, 1),
                                       (1, 68, 120, 64, 64, 3, 1),
                                       (1, 68, 120, 64, 32, 3, 1),
                                       (1, 256, 256, 3, 16, 3, 1),
                                       (1, 64, 64, 128, 128, 3, 1),
                                       (1, 256, 256, 32, 32, 1, 1)]:
        pad = [1, 1, 1, 1] if k == 3 and stride == 1 else ([1, 0, 1, 0] if k == 3 else [0, 0, 0, 0])
        g = add(Graph(f'shape_conv2d_i8_rescale_{n}x{h}x{w}x{ic}_oc{oc}_k{k}_s{stride}'))
        x = g.input('i8', [n, h, w, ic])
        r, dims = conv2d_i8(g, rng, x, n, h, w, ic, oc, k, pad, [stride, stride], [1, 1])
        g.output(rescale_after(g, rng, r, dims, RESCALE_VARIANTS[0]), 'i8', dims)


def gen_conv_zero_points():
    rng = random.Random(3333)
    cases = [
        (1, 5, 6, 3, 4, [1, 1, 1, 1], -128, 0),
        (2, 7, 5, 5, 3, [0, 0, 0, 0], 7, -3),
        (3, 6, 9, 2, 5, [2, 1, 0, 2], 0, 127),
        (1, 4, 4, 7, 2, [1, 0, 1, 0], -128, -128),
        (2, 9, 8, 8, 8, [0, 1, 1, 0], 127, 5),
        (1, 1, 1, 1, 1, [1, 1, 1, 1], 0, 0),
        (1, 3, 3, 4, 6, [0, 0, 0, 0], -128, 0),
        (1, 2, 7, 9, 7, [2, 2, 0, 1], 3, -1),
        (1, 16, 15, 17, 13, [1, 1, 1, 1], -128, 0),
        (2, 11, 12, 16, 16, [1, 1, 1, 1], 0, 0),
        (1, 12, 10, 1, 9, [1, 1, 1, 1], -5, 100),
    ]

    def tag(n, h, w, ic, oc, pad, izp, wzp):
        return f'{n}x{h}x{w}x{ic}_oc{oc}_p{"".join(map(str, pad))}_z{izp}_w{wzp}'

    for case in cases:
        n, h, w, ic, oc, pad, izp, wzp = case
        g = add(Graph(f'conv3x3_i8_{tag(*case)}'))
        x = g.input('i8', [n, h, w, ic])
        r, dims = conv2d_i8(g, rng, x, n, h, w, ic, oc, 3, pad, [1, 1], [1, 1], izp=izp, wzp=wzp)
        g.output(r, 'i32', dims)

    for case in (cases[0], cases[4], cases[8]):
        n, h, w, ic, oc, pad, izp, wzp = case
        for vi, variant in enumerate(RESCALE_VARIANTS):
            g = add(Graph(f'conv3x3_i8_rescale_v{vi}_{tag(*case)}'))
            x = g.input('i8', [n, h, w, ic])
            r, dims = conv2d_i8(g, rng, x, n, h, w, ic, oc, 3, pad, [1, 1], [1, 1], izp=izp, wzp=wzp)
            g.output(rescale_after(g, rng, r, dims, variant), variant[5], dims)

    for ic in (900, 1024):
        g = add(Graph(f'conv3x3_i8_extreme_1x3x3x{ic}'))
        x = g.input('i8', [1, 3, 3, ic])
        low = g.m.op('CLAMP', 'i8', [1, 3, 3, ic], g.m.scalar('i8', -128), g.m.scalar('i8', -128), g.m.uint(1), x)
        weights = g.m.const('i8', [1, 3, 3, ic], [127] * (9 * ic))
        bias = g.m.const('i32', [1], [3])
        g.output(g.m.op('CONV2D', 'i32', [1, 1, 1, 1], g.m.const('i32', [4], [0, 0, 0, 0]), g.m.const('i32', [2], [1, 1]),
                        g.m.const('i32', [2], [1, 1]), g.m.uint(1), g.m.boolc(False), low, weights, bias,
                        zp(g, 'i8', 127), zp(g, 'i8', -128)), 'i32', [1, 1, 1, 1])


def gen_conv_decomp():
    rng = random.Random(777)
    for h, w in [(64, 64), (68, 120), (128, 128)]:
        for c in (32, 64, 128):
            g = add(Graph(f'decomp_conv2d_i8_1x{h}x{w}x{c}_oc{c}'))
            x = g.input('i8', [1, h, w, c])
            r, dims = conv2d_i8(g, rng, x, 1, h, w, c, c, 3, [1, 1, 1, 1], [1, 1], [1, 1])
            g.output(r, 'i32', dims)
    for h, w, ic, oc in [(256, 256, 3, 16), (544, 960, 16, 4), (272, 480, 48, 16), (68, 120, 64, 32), (136, 240, 32, 36)]:
        g = add(Graph(f'decomp_conv2d_i8_1x{h}x{w}x{ic}_oc{oc}'))
        x = g.input('i8', [1, h, w, ic])
        r, dims = conv2d_i8(g, rng, x, 1, h, w, ic, oc, 3, [1, 1, 1, 1], [1, 1], [1, 1])
        g.output(r, 'i32', dims)


def gen_conv_mid():
    rng = random.Random(778)
    for h, w, ic, oc in [(64, 64, 128, 128), (68, 120, 64, 64), (68, 120, 128, 128), (128, 128, 64, 64),
                         (544, 960, 16, 4), (256, 256, 3, 16)]:
        g = add(Graph(f'mid_conv2d_i8_1x{h}x{w}x{ic}_oc{oc}'))
        x = g.input('i8', [1, h, w, ic])
        mid = g.m.op('CLAMP', 'i8', [1, h, w, ic], g.m.scalar('i8', -128), g.m.scalar('i8', 127), g.m.uint(1), x)
        r, dims = conv2d_i8(g, rng, mid, 1, h, w, ic, oc, 3, [1, 1, 1, 1], [1, 1], [1, 1])
        g.output(r, 'i32', dims)


def gen_conv_tiles():
    rng = random.Random(8888)
    cases = [
        (2, 13, 11, 7, 6, 3, 5, [2, 0, 1, 3], [2, 1], [1, 3], 5, -7),
        (3, 9, 17, 5, 9, 1, 1, [0, 0, 0, 0], [2, 2], [1, 1], -128, 3),
        (1, 20, 20, 12, 5, 7, 7, [3, 2, 3, 2], [3, 3], [2, 2], 0, 1),
        (2, 6, 7, 3, 4, 3, 3, [4, 4, 4, 4], [1, 1], [1, 1], 7, 0),
        (1, 13, 11, 256, 3, 3, 3, [1, 1, 1, 1], [2, 2], [1, 1], -128, 0),
        (1, 40, 38, 16, 8, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1], 127, -128),
        (1, 1, 1, 4, 5, 1, 1, [0, 0, 0, 0], [1, 1], [1, 1], 0, 0),
        (2, 23, 20, 33, 11, 3, 3, [1, 1, 1, 0], [2, 2], [1, 1], -3, 2),
    ]

    def conv(g, x, case):
        n, h, w, ic, oc, kh, kw, pad, stride, dil, izp, wzp = case
        assert (h - 1 + pad[0] + pad[1] - (kh - 1) * dil[0]) % stride[0] == 0, case
        assert (w - 1 + pad[2] + pad[3] - (kw - 1) * dil[1]) % stride[1] == 0, case
        oh = (h - 1 + pad[0] + pad[1] - (kh - 1) * dil[0]) // stride[0] + 1
        ow = (w - 1 + pad[2] + pad[3] - (kw - 1) * dil[1]) // stride[1] + 1
        weights = g.m.const('i8', [oc, kh, kw, ic], [rng.randrange(-128, 128) for _ in range(oc * kh * kw * ic)])
        bias = g.m.const('i32', [oc], [rng.randrange(-50000, 50000) for _ in range(oc)])
        dims = [n, oh, ow, oc]
        r = g.m.op('CONV2D', 'i32', dims, g.m.const('i32', [4], pad), g.m.const('i32', [2], stride), g.m.const('i32', [2], dil),
                   g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', izp), zp(g, 'i8', wzp))
        return r, dims

    def tag(case):
        n, h, w, ic, oc, kh, kw, pad, stride, dil, izp, wzp = case
        return (f'{n}x{h}x{w}x{ic}_oc{oc}_k{kh}{kw}_s{stride[0]}{stride[1]}_d{dil[0]}{dil[1]}_p{"".join(map(str, pad))}'
                f'_z{izp}_w{wzp}')

    for case in cases:
        n, h, w, ic = case[:4]
        g = add(Graph(f'tile_i8_{tag(case)}'))
        x = g.input('i8', [n, h, w, ic])
        r, dims = conv(g, x, case)
        g.output(r, 'i32', dims)

    for case in (cases[0], cases[5], cases[7]):
        n, h, w, ic = case[:4]
        for vi, variant in enumerate(RESCALE_VARIANTS):
            g = add(Graph(f'tile_i8_rescale_v{vi}_{tag(case)}'))
            x = g.input('i8', [n, h, w, ic])
            r, dims = conv(g, x, case)
            g.output(rescale_after(g, rng, r, dims, variant), variant[5], dims)


def gen_resize():
    cases = [
        (1, 9, 11, 3),
        (2, 16, 13, 5),
        (1, 7, 5, 16),
        (3, 8, 8, 17),
        (1, 5, 6, 1),
        (1, 12, 9, 32),
    ]
    settings = [
        ([2, 1, 2, 1], [0, 0], [0, 0]),
        ([4, 2, 4, 2], [-1, -1], [1, 1]),
        ([1, 2, 1, 2], [0, 0], [0, 0]),
        ([3, 5, 5, 3], [0, 1], [0, 0]),
        ([5, 2, 3, 2], [-2, 1], [1, 0]),
    ]
    for n, h, w, c in cases:
        for mode in (1, 2):
            for scale, offset, border in settings:
                oh = ((h - 1) * scale[0] - offset[0] + border[0]) // scale[1] + 1
                ow = ((w - 1) * scale[2] - offset[1] + border[1]) // scale[3] + 1
                if oh < 1 or ow < 1:
                    continue
                tag = f'{n}x{h}x{w}x{c}_m{mode}_s{"_".join(map(str, scale))}_o{offset[0]}_{offset[1]}'
                for element, out_e in (('i8', 'i8' if mode == 1 else 'i32'), ('f32', 'f32'), ('f16', 'f16')):
                    g = add(Graph(f'resize_{element}_{tag}'))
                    a = g.input(element, [n, h, w, c])
                    g.output(g.m.op('RESIZE', out_e, [n, oh, ow, c], g.m.uint(mode), a, g.m.const('i32', [4], scale),
                                    g.m.const('i32', [2], offset), g.m.const('i32', [2], border)), out_e, [n, oh, ow, c])


def gen_concat():
    for axis, shapes in [
        (3, [[1, 5, 7, 3], [1, 5, 7, 5]]),
        (3, [[2, 4, 6, 16], [2, 4, 6, 17]]),
        (3, [[1, 9, 9, 8], [1, 9, 9, 8], [1, 9, 9, 8]]),
        (0, [[1, 5, 7, 6], [2, 5, 7, 6]]),
        (1, [[1, 3, 7, 6], [1, 9, 7, 6]]),
        (2, [[2, 5, 4, 12], [2, 5, 11, 12]]),
        (1, [[3, 5], [7, 5]]),
        (2, [[1, 2, 3, 4, 5], [1, 2, 6, 4, 5]]),
    ]:
        dims = list(shapes[0])
        dims[axis] = sum(shape[axis] for shape in shapes)
        tag = f'ax{axis}_{len(shapes)}x{"x".join(map(str, shapes[0]))}'
        for element in ('i8', 'i32', 'f32'):
            g = add(Graph(f'concat_{element}_{tag}'))
            inputs = [g.input(element, shape) for shape in shapes]
            g.output(g.m.op('CONCAT', element, dims, g.m.uint(axis), *inputs), element, dims)


def gen_pool():
    cases = [(1, 9, 11, 3), (2, 16, 13, 6), (1, 8, 8, 16), (1, 7, 9, 17), (1, 5, 6, 1), (2, 12, 10, 32)]
    windows = [([2, 2], [2, 2], [0, 0, 0, 0]), ([3, 3], [1, 1], [1, 1, 1, 1]), ([3, 2], [2, 1], [0, 1, 1, 0]),
               ([1, 1], [1, 1], [0, 0, 0, 0]), ([5, 3], [1, 2], [2, 2, 1, 1])]
    first = True
    for n, h, w, c in cases:
        for k, s, p in windows:
            oh = (h + p[0] + p[1] - k[0]) // s[0] + 1
            ow = (w + p[2] + p[3] - k[1]) // s[1] + 1
            if oh < 1 or ow < 1:
                continue
            dims = [n, oh, ow, c]
            tag = f'{n}x{h}x{w}x{c}_k{k[0]}{k[1]}_s{s[0]}{s[1]}_p{"".join(map(str, p))}'

            g = add(Graph(f'pool_max_i8_{tag}'))
            a = g.input('i8', [n, h, w, c])
            g.output(g.m.op('MAX_POOL2D', 'i8', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                            g.m.const('i32', [4], p), g.m.uint(1), a), 'i8', dims)

            g = add(Graph(f'pool_avg_i8_{tag}'))
            a = g.input('i8', [n, h, w, c])
            g.output(g.m.op('AVG_POOL2D', 'i8', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                            g.m.const('i32', [4], p), g.m.uint(1), a, zp(g, 'i8', -3), zp(g, 'i8', 5)), 'i8', dims)

            g = add(Graph(f'pool_max_f32_{tag}'))
            a = g.input('f32', [n, h, w, c])
            g.output(g.m.op('MAX_POOL2D', 'f32', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                            g.m.const('i32', [4], p), g.m.uint(1), a), 'f32', dims)

            g = add(Graph(f'pool_avg_f32_{tag}'))
            a = g.input('f32', [n, h, w, c])
            g.output(g.m.op('AVG_POOL2D', 'f32', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                            g.m.const('i32', [4], p), g.m.uint(3), a, zp(g, 'f32', 0.0), zp(g, 'f32', 0.0)), 'f32', dims)

            if first:
                g = add(Graph(f'pool_max_f32_ignore_{tag}'))
                a = g.input('f32', [n, h, w, c])
                g.output(g.m.op('MAX_POOL2D', 'f32', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                                g.m.const('i32', [4], p), g.m.uint(2), a), 'f32', dims)

                g = add(Graph(f'pool_max_f16_{tag}'))
                a = g.input('f16', [n, h, w, c])
                g.output(g.m.op('MAX_POOL2D', 'f16', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                                g.m.const('i32', [4], p), g.m.uint(1), a), 'f16', dims)

                g = add(Graph(f'pool_avg_f16_{tag}'))
                a = g.input('f16', [n, h, w, c])
                g.output(g.m.op('AVG_POOL2D', 'f16', dims, g.m.const('i32', [2], k), g.m.const('i32', [2], s),
                                g.m.const('i32', [4], p), g.m.uint(3), a, zp(g, 'f16', 0.0), zp(g, 'f16', 0.0)),
                         'f16', dims)
        first = False


def gen_conv_float():
    rng = random.Random(9001)
    cases = [
        (1, 8, 9, 3, 5, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
        (1, 9, 10, 16, 5, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
        (1, 8, 12, 32, 32, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
        (2, 11, 9, 17, 7, 3, 3, [0, 1, 1, 0], [2, 2], [1, 1]),
        (1, 12, 11, 6, 4, 5, 3, [2, 2, 1, 1], [1, 2], [1, 1]),
        (1, 13, 14, 4, 9, 3, 3, [2, 2, 2, 2], [1, 1], [2, 2]),
        (1, 7, 9, 48, 16, 1, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
        (1, 5, 5, 1, 1, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1]),
    ]
    pool = [1.0 / 8, -1.0 / 4, 1024.0, -2048.0, 1.0 / 4096, -1.0 / 8192, 0.0, -0.0, 3.5, -7.25]

    def value():
        return pool[rng.randrange(len(pool))]

    for n, h, w, ic, oc, kh, kw, pad, stride, dil in cases:
        oh = conv_out(h, pad[0], pad[1], kh, stride[0], dil[0])
        ow = conv_out(w, pad[2], pad[3], kw, stride[1], dil[1])
        tag = f'{n}x{h}x{w}x{ic}_oc{oc}_k{kh}{kw}_s{stride[0]}{stride[1]}_d{dil[0]}_p{"".join(map(str, pad))}'
        for element, acc in (('f32', 3), ('f16', 2), ('f16', 3)):
            g = add(Graph(f'convf_{element}_a{acc}_{tag}'))
            x = g.input(element, [n, h, w, ic])
            weights = g.m.const(element, [oc, kh, kw, ic], [value() for _ in range(oc * kh * kw * ic)])
            bias = g.m.const(element, [oc], [value() for _ in range(oc)])
            g.output(g.m.op('CONV2D', element, [n, oh, ow, oc], g.m.const('i32', [4], pad),
                            g.m.const('i32', [2], stride), g.m.const('i32', [2], dil), g.m.uint(acc),
                            g.m.boolc(False), x, weights, bias, zp(g, element, 0.0), zp(g, element, 0.0)),
                     element, [n, oh, ow, oc])

        g = add(Graph(f'convf_f32_bias1_{tag}'))
        x = g.input('f32', [n, h, w, ic])
        weights = g.m.const('f32', [oc, kh, kw, ic], [value() for _ in range(oc * kh * kw * ic)])
        g.output(g.m.op('CONV2D', 'f32', [n, oh, ow, oc], g.m.const('i32', [4], pad), g.m.const('i32', [2], stride),
                        g.m.const('i32', [2], dil), g.m.uint(3), g.m.boolc(False), x, weights,
                        g.m.const('f32', [1], [value()]), zp(g, 'f32', 0.0), zp(g, 'f32', 0.0)),
                 'f32', [n, oh, ow, oc])


def gen_conv_lanes():
    rng = random.Random(4242)
    pool = [0.1, -0.3, 1.0 / 3, -1.0 / 7, 2.7, 1024.1, -0.001, 0.0, -0.0, 5.0e-5]

    def weights_of(element, dims):
        if element == 'i8':
            return g.m.const('i8', dims, [rng.randrange(-128, 128) for _ in range(prod(dims))])
        return g.m.const(element, dims, [pool[rng.randrange(len(pool))] for _ in range(prod(dims))])

    def bias_of(element, count):
        if element == 'i8':
            return g.m.const('i32', [count], [rng.randrange(-5000, 5000) for _ in range(count)])
        return g.m.const(element, [count], [pool[rng.randrange(len(pool))] for _ in range(count)])

    kinds = [('i8', 'i32', 1, -128), ('f32', 'f32', 3, 0.0), ('f16', 'f16', 2, 0.0)]

    for n, d, h, w, ic, oc in [(1, 4, 5, 6, 6, 5), (2, 3, 4, 5, 9, 2), (1, 3, 3, 4, 4, 7)]:
        od, oh, ow = conv_out(d, 1, 1, 3, 1, 1), conv_out(h, 1, 0, 3, 2, 1), conv_out(w, 0, 1, 3, 1, 1)
        for element, out_e, acc, izp in kinds:
            g = add(Graph(f'lanes_conv3d_{element}_{n}x{d}x{h}x{w}x{ic}_oc{oc}'))
            x = g.input(element, [n, d, h, w, ic])
            g.output(g.m.op('CONV3D', out_e, [n, od, oh, ow, oc], g.m.const('i32', [6], [1, 1, 1, 0, 0, 1]),
                            g.m.const('i32', [3], [1, 2, 1]), g.m.const('i32', [3], [1, 1, 1]), g.m.uint(acc),
                            g.m.boolc(False), x, weights_of(element, [oc, 3, 3, 3, ic]), bias_of(element, oc),
                            zp(g, element, izp), zp(g, element, 0)), out_e, [n, od, oh, ow, oc])

    for n, h, w, c, m, stride in [(1, 9, 8, 5, 2, 1), (1, 10, 11, 4, 1, 2), (2, 7, 7, 3, 3, 1), (1, 8, 8, 6, 1, 1)]:
        oh, ow = conv_out(h, 1, 1, 3, stride, 1), conv_out(w, 1, 0, 3, stride, 1)
        for element, out_e, acc, izp in kinds:
            g = add(Graph(f'lanes_dwconv2d_{element}_{n}x{h}x{w}x{c}_m{m}_s{stride}'))
            x = g.input(element, [n, h, w, c])
            g.output(g.m.op('DEPTHWISE_CONV2D', out_e, [n, oh, ow, c * m], g.m.const('i32', [4], [1, 1, 1, 0]),
                            g.m.const('i32', [2], [stride, stride]), g.m.const('i32', [2], [1, 1]), g.m.uint(acc),
                            g.m.boolc(False), x, weights_of(element, [3, 3, c, m]), bias_of(element, c * m),
                            zp(g, element, izp), zp(g, element, 0)), out_e, [n, oh, ow, c * m])

    for n, h, w, ic, oc, k, out_pad, stride in [(1, 5, 6, 3, 5, 3, [0, 0, 0, 0], [2, 2]), (1, 4, 4, 6, 3, 4, [-1, -1, -1, -1], [2, 2]),
                                                (2, 3, 5, 5, 6, 3, [0, 1, 1, 0], [1, 3])]:
        oh = (h - 1) * stride[0] + out_pad[0] + out_pad[1] + k
        ow = (w - 1) * stride[1] + out_pad[2] + out_pad[3] + k
        for element, out_e, acc, izp in kinds:
            g = add(Graph(f'lanes_tconv2d_{element}_{n}x{h}x{w}x{ic}_oc{oc}_k{k}_s{stride[0]}{stride[1]}'))
            x = g.input(element, [n, h, w, ic])
            g.output(g.m.op('TRANSPOSE_CONV2D', out_e, [n, oh, ow, oc], g.m.const('i32', [4], out_pad),
                            g.m.const('i32', [2], stride), g.m.uint(acc), g.m.boolc(False), x,
                            weights_of(element, [oc, k, k, ic]), bias_of(element, oc), zp(g, element, izp),
                            zp(g, element, 0)), out_e, [n, oh, ow, oc])

    for n, h, c, w in [(1, 5, 7, 3), (2, 9, 13, 11), (1, 4, 6, 5)]:
        for element in ['f32', 'f16']:
            g = add(Graph(f'lanes_matmul_{element}_{n}x{h}x{c}x{w}'))
            a = g.input(element, [n, h, c])
            b = g.m.op('MUL', element, [n, c, w], g.input(element, [n, c, w]), g.m.const(element, [1, 1, 1], [0.3]),
                       g.m.const('i8', [1], [0]))
            g.output(g.m.op('MATMUL', element, [n, h, w], a, b, zp(g, element, 0.0), zp(g, element, 0.0)), element, [n, h, w])


def gen_tconv_taps():
    def taps(n, h, w, ic, oc, k, out_pad, stride, oh, ow):
        counts = [0] * (n * oh * ow * oc)
        for b in range(n):
            for iy in range(h):
                for ix in range(w):
                    for ky in range(k):
                        for kx in range(k):
                            oy = iy * stride[0] + out_pad[0] + ky
                            ox = ix * stride[1] + out_pad[2] + kx
                            if 0 <= oy < oh and 0 <= ox < ow:
                                for c in range(oc):
                                    counts[((b * oh + oy) * ow + ox) * oc + c] += ic
        return counts

    for n, h, w, ic, oc, k, out_pad, stride in [(1, 3, 5, 2, 3, 3, [0, 1, 1, 0], [1, 3]),
                                                (1, 4, 4, 2, 2, 3, [1, 0, 0, 1], [2, 2]),
                                                (1, 3, 3, 3, 2, 4, [-1, -1, -1, -1], [2, 3]),
                                                (2, 2, 3, 1, 2, 2, [2, 0, 1, 2], [3, 1])]:
        oh = (h - 1) * stride[0] + out_pad[0] + out_pad[1] + k
        ow = (w - 1) * stride[1] + out_pad[2] + out_pad[3] + k
        dims = [n, oh, ow, oc]
        g = add(Graph(f'taps_tconv2d_i8_{n}x{h}x{w}x{ic}_oc{oc}_k{k}_s{stride[0]}{stride[1]}'
                      f'_op{"m" if min(out_pad) < 0 else ""}{"".join(str(abs(v)) for v in out_pad)}'))
        x = g.input('i8', [n, h, w, ic])
        ones = g.m.op('CLAMP', 'i8', [n, h, w, ic], g.m.scalar('i8', 1), g.m.scalar('i8', 1), g.m.uint(1), x)
        weights = g.m.const('i8', [oc, k, k, ic], [1] * (oc * k * k * ic))
        bias = g.m.const('i32', [oc], [0] * oc)
        g.output(g.m.op('TRANSPOSE_CONV2D', 'i32', dims, g.m.const('i32', [4], out_pad),
                        g.m.const('i32', [2], stride), g.m.uint(1), g.m.boolc(False), ones, weights, bias,
                        zp(g, 'i8', 0), zp(g, 'i8', 0)), 'i32', dims)
        REFERENCES[g.name] = taps(n, h, w, ic, oc, k, out_pad, stride, oh, ow)


def gen_table():
    t8 = [((i * 37) % 256) - 128 for i in range(256)]
    t8edge = [(127 if i % 2 == 0 else -128) for i in range(256)]
    t16 = [((i * 7919) % 65536) - 32768 for i in range(513)]
    for dims in [[7], [3, 5], [2, 3, 4, 5], [1, 5, 7, 3], [2, 4, 6, 16], [3, 8, 8, 17], [1, 2, 3, 4, 5], [2, 33, 65, 4]]:
        tag = 'x'.join(map(str, dims))
        g = add(Graph(f'table_i8_{tag}'))
        a = g.input('i8', dims)
        g.output(g.m.op('TABLE', 'i8', dims, a, g.m.const('i8', [256], t8)), 'i8', dims)

        g = add(Graph(f'table_i8_edge_{tag}'))
        a = g.input('i8', dims)
        g.output(g.m.op('TABLE', 'i8', dims, a, g.m.const('i8', [256], t8edge)), 'i8', dims)

        g = add(Graph(f'table_i16_{tag}'))
        a = g.input('i16', dims)
        g.output(g.m.op('TABLE', 'i32', dims, a, g.m.const('i16', [513], t16)), 'i32', dims)


def gen_operators():
    gen_unary()
    gen_binary()
    gen_rescale()
    gen_layout()
    gen_spatial()
    gen_conv()

def rescale_from(g, x, dims, mult, shift, dst='i8', per_channel=True, scale32=True, rounding=3, izp=0, ozp=-128):
    return g.m.op('RESCALE', dst, dims, g.m.boolc(scale32), g.m.uint(rounding), g.m.boolc(per_channel), g.m.boolc(False),
                  g.m.boolc(False), x, mult, shift, zp(g, 'i32', izp), zp(g, dst, ozp))


def mult_values(rng, n, scale32=True):
    if scale32:
        return [rng.randrange(1 << 29, (1 << 31) - 1) for _ in range(n)]
    return [rng.randrange(1 << 13, (1 << 15) - 1) for _ in range(n)]


def shift_values(rng, n, scale32=True):
    return [rng.randrange(36, 46) if scale32 else rng.randrange(20, 30) for _ in range(n)]


def gen_fusion_order():
    rng = random.Random(77)
    conv = (1, 9, 10, 3, 5, 3, [1, 1, 1, 1], [1, 1], [1, 1])
    n, h, w, ic, oc = conv[:5]

    g = add(Graph('order_conv2d_mult_cast_after'))
    x = g.input('i8', [n, h, w, ic])
    r, dims = conv2d_i8(g, rng, x, *conv)
    mult = g.m.op('CAST', 'i32', [oc], g.m.const('i16', [oc], [v >> 16 for v in mult_values(rng, oc)]))
    shift = g.m.const('i8', [oc], shift_values(rng, oc))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_conv2d_shift_cast_after'))
    x = g.input('i8', [n, h, w, ic])
    r, dims = conv2d_i8(g, rng, x, *conv)
    mult = g.m.const('i32', [oc], mult_values(rng, oc))
    shift = g.m.op('CAST', 'i8', [oc], g.m.const('i16', [oc], shift_values(rng, oc)))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_conv2d_mult_from_input_after'))
    x = g.input('i8', [n, h, w, ic])
    m = g.input('i8', [oc])
    r, dims = conv2d_i8(g, rng, x, *conv)
    mult = g.m.op('ADD', 'i32', [oc], g.m.op('CAST', 'i32', [oc], m), g.m.const('i32', [oc], mult_values(rng, oc)))
    shift = g.m.const('i8', [oc], shift_values(rng, oc))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_matmul_mult_cast_after'))
    a, b = g.input('i8', [1, 5, 7]), g.input('i8', [1, 7, 3])
    r = g.m.op('MATMUL', 'i32', [1, 5, 3], a, b, zp(g, 'i8', -128), zp(g, 'i8', 3))
    mult = g.m.op('CAST', 'i32', [3], g.m.const('i16', [3], [v >> 16 for v in mult_values(rng, 3)]))
    shift = g.m.const('i8', [3], shift_values(rng, 3))
    g.output(rescale_from(g, r, [1, 5, 3], mult, shift), 'i8', [1, 5, 3])

    g = add(Graph('order_dwconv2d_mult_cast_after'))
    c, m_ = 3, 2
    x = g.input('i8', [1, 9, 8, c])
    oh, ow = conv_out(9, 1, 1, 3, 1, 1), conv_out(8, 1, 1, 3, 1, 1)
    weights = g.m.const('i8', [3, 3, c, m_], [rng.randrange(-128, 128) for _ in range(9 * c * m_)])
    bias = g.m.const('i32', [c * m_], [rng.randrange(-50000, 50000) for _ in range(c * m_)])
    dims = [1, oh, ow, c * m_]
    r = g.m.op('DEPTHWISE_CONV2D', 'i32', dims, g.m.const('i32', [4], [1, 1, 1, 1]), g.m.const('i32', [2], [1, 1]),
               g.m.const('i32', [2], [1, 1]), g.m.uint(1), g.m.boolc(False), x, weights, bias, zp(g, 'i8', -128),
               zp(g, 'i8', 0))
    mult = g.m.op('CAST', 'i32', [c * m_], g.m.const('i16', [c * m_], [v >> 16 for v in mult_values(rng, c * m_)]))
    shift = g.m.const('i8', [c * m_], shift_values(rng, c * m_))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_conv2d_mult_cast_before'))
    x = g.input('i8', [n, h, w, ic])
    mult = g.m.op('CAST', 'i32', [oc], g.m.const('i16', [oc], [v >> 16 for v in mult_values(rng, oc)]))
    r, dims = conv2d_i8(g, rng, x, *conv)
    shift = g.m.const('i8', [oc], shift_values(rng, oc))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_conv2d_mult_from_input_before'))
    x = g.input('i8', [n, h, w, ic])
    m = g.input('i8', [oc])
    mult = g.m.op('ADD', 'i32', [oc], g.m.op('CAST', 'i32', [oc], m), g.m.const('i32', [oc], mult_values(rng, oc)))
    r, dims = conv2d_i8(g, rng, x, *conv)
    shift = g.m.const('i8', [oc], shift_values(rng, oc))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_conv2d_mult_is_input'))
    x = g.input('i8', [n, h, w, ic])
    mult = g.input('i32', [oc])
    r, dims = conv2d_i8(g, rng, x, *conv)
    shift = g.m.const('i8', [oc], shift_values(rng, oc))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)

    g = add(Graph('order_conv2d_mult_is_output_after'))
    x = g.input('i8', [n, h, w, ic])
    m = g.input('i8', [oc])
    r, dims = conv2d_i8(g, rng, x, *conv)
    mult = g.m.op('ADD', 'i32', [oc], g.m.op('CAST', 'i32', [oc], m), g.m.const('i32', [oc], mult_values(rng, oc)))
    shift = g.m.const('i8', [oc], shift_values(rng, oc))
    g.output(rescale_from(g, r, dims, mult, shift), 'i8', dims)
    g.output(mult, 'i32', [oc])


BIG = [4, 512, 512, 16]


def gen_bench_narrow():
    rng = random.Random(20260918)

    g = add(Graph(f'bench_mul_i8_{'x'.join(map(str, BIG))}'))
    a, b = g.input('i8', BIG), g.input('i8', BIG)
    g.output(g.m.op('MUL', 'i8', BIG, a, b, g.m.const('i8', [1], [2])), 'i8', BIG)

    g = add(Graph(f'bench_mul_i16_shift_{'x'.join(map(str, BIG))}'))
    a, b = g.input('i16', BIG), g.input('i16', BIG)
    g.output(g.m.op('MUL', 'i16', BIG, a, b, g.m.const('i8', [1], [6])), 'i16', BIG)

    g = add(Graph(f'bench_select_i8_{'x'.join(map(str, BIG))}'))
    a, b = g.input('i8', BIG), g.input('i8', BIG)
    mask = g.m.op('GREATER_EQUAL', 'bool', BIG, a, b)
    g.output(g.m.op('SELECT', 'i8', BIG, mask, a, b), 'i8', BIG)

    g = add(Graph(f'bench_arithmetic_right_shift_i8_{'x'.join(map(str, BIG))}'))
    a = g.input('i8', BIG)
    shifts = g.m.const('i8', [1, 1, 1, 16], [(i * 7) % 8 for i in range(16)])
    g.output(g.m.op('ARITHMETIC_RIGHT_SHIFT', 'i8', BIG, g.m.boolc(True), a, shifts), 'i8', BIG)

    g = add(Graph(f'bench_rescale_i8_i8_{'x'.join(map(str, BIG))}'))
    a = g.input('i8', BIG)
    mult = g.m.const('i16', [16], [rng.randrange(1 << 13, (1 << 15) - 1) for _ in range(16)])
    shift = g.m.const('i8', [16], [rng.randrange(14, 18) for _ in range(16)])
    g.output(g.m.op('RESCALE', 'i8', BIG, g.m.boolc(False), g.m.uint(3), g.m.boolc(True), g.m.boolc(False),
                    g.m.boolc(False), a, mult, shift, zp(g, 'i8', 0), zp(g, 'i8', 0)), 'i8', BIG)

    n, k, c, w = 4, 4096, 64, 64
    g = add(Graph(f'bench_gather_i8_{n}x{k}x{c}'))
    values = g.input('i8', [n, k, c])
    idx = g.m.const('i32', [n, k], [rng.randrange(k) for _ in range(n * k)])
    g.output(g.m.op('GATHER', 'i8', [n, k, c], values, idx), 'i8', [n, k, c])

    g = add(Graph(f'bench_scatter_i8_{n}x{k}x{c}'))
    vin = g.input('i8', [n, k, c])
    upd = g.input('i8', [n, w, c], seed=77)
    idxs = []
    for _ in range(n):
        idxs += rng.sample(range(k), w)
    g.output(g.m.op('SCATTER', 'i8', [n, k, c], vin, g.m.const('i32', [n, w], idxs), upd), 'i8', [n, k, c])


def gen_bench_ops():
    rng = random.Random(20260919)

    n, d, h, w, ic, oc, k = 1, 8, 64, 64, 16, 16, 3
    od, oh, ow = conv_out(d, 1, 1, k, 1, 1), conv_out(h, 1, 1, k, 1, 1), conv_out(w, 1, 1, k, 1, 1)
    g = add(Graph(f'bench_conv3d_i8_{n}x{d}x{h}x{w}x{ic}_oc{oc}_k{k}'))
    a = g.input('i8', [n, d, h, w, ic])
    weights = g.m.const('i8', [oc, k, k, k, ic], [rng.randrange(-8, 9) for _ in range(oc * k * k * k * ic)])
    bias = g.m.const('i32', [oc], [rng.randrange(-5000, 5000) for _ in range(oc)])
    g.output(g.m.op('CONV3D', 'i32', [n, od, oh, ow, oc], g.m.const('i32', [6], [1] * 6),
                    g.m.const('i32', [3], [1] * 3), g.m.const('i32', [3], [1] * 3), g.m.uint(1), g.m.boolc(False),
                    a, weights, bias, zp(g, 'i8', 0), zp(g, 'i8', 0)), 'i32', [n, od, oh, ow, oc])

    flat = [4, 512, 8192]
    g = add(Graph(f'bench_reshape_i8_{'x'.join(map(str, BIG))}'))
    a = g.input('i8', BIG)
    g.output(g.m.op('RESHAPE', 'i8', flat, a, g.m.const('i32', [len(flat)], flat)), 'i8', flat)

    out1 = [4, 1, 512, 16]
    g = add(Graph(f'bench_reduce_sum_i32_ax1_{'x'.join(map(str, BIG))}'))
    a = g.input('i32', BIG)
    g.output(g.m.op('REDUCE_SUM', 'i32', out1, g.m.uint(1), a), 'i32', out1)

    g = add(Graph(f'bench_reduce_max_i8_ax1_{'x'.join(map(str, BIG))}'))
    a = g.input('i8', BIG)
    g.output(g.m.op('REDUCE_MAX', 'i8', out1, g.m.uint(1), g.m.uint(1), a), 'i8', out1)

    g = add(Graph(f'bench_argmax_i8_ax1_{'x'.join(map(str, BIG))}'))
    a = g.input('i8', BIG)
    g.output(g.m.op('ARGMAX', 'i32', [4, 512, 16], g.m.uint(1), g.m.uint(1), a), 'i32', [4, 512, 16])


def fft2d(n, h, w, inverse):
    g = add(Graph(f'fft2d_f32_{n}x{h}x{w}{"_inv" if inverse else ""}'))
    real = g.input('f32', [n, h, w])
    imag = g.input('f32', [n, h, w])
    outReal, outImag = g.m.op2('FFT2D', 'f32', [n, h, w], g.m.boolc(inverse), g.m.boolc(False), real, imag)
    g.output(outReal, 'f32', [n, h, w])
    g.output(outImag, 'f32', [n, h, w])


def rfft2d(n, h, w):
    out = [n, h, w // 2 + 1]
    g = add(Graph(f'rfft2d_f32_{n}x{h}x{w}'))
    real = g.input('f32', [n, h, w])
    outReal, outImag = g.m.op2('RFFT2D', 'f32', out, g.m.boolc(False), real)
    g.output(outReal, 'f32', out)
    g.output(outImag, 'f32', out)


def gen_fft():
    for n, h, w in ((1, 8, 8), (2, 4, 16), (1, 16, 4), (1, 8, 16), (3, 4, 4)):
        fft2d(n, h, w, False)
        fft2d(n, h, w, True)
        rfft2d(n, h, w)
    fft2d(4, 64, 64, False)
    rfft2d(4, 64, 64)
    fft2d(16, 64, 64, False)
    rfft2d(16, 64, 64)
    fft2d(1, 128, 128, False)
    rfft2d(1, 128, 128)


SUITES = {
    'operators': (
        'Every operator family over ranks 1 to 6, last dimensions on and off multiples of four, broadcast '
        'patterns with a last dimension of one on either side, tensors larger than one dispatch row, the '
        'rounding modes and zero points of RESCALE, and small CONV2D, DEPTHWISE_CONV2D, TRANSPOSE_CONV2D, '
        'CONV3D and MATMUL cases in int8 and float32.',
        gen_operators,
    ),
    'fusion': (
        'Accumulating operators followed by every RESCALE variant, and the cases where nothing may be merged: '
        'the intermediate tensor is a graph output, it has two consumers, the RESCALE output is unsigned; '
        'also a chain of two merged convolutions with a CLAMP behind the last RESCALE.',
        gen_fusion,
    ),
    'fusion_order': (
        'RESCALE multipliers and shifts produced by an operator that follows the convolution in program '
        'order, where a merged kernel would read them before they are written, also when that multiplier is '
        'a graph output as well, and the controls where the producer precedes the convolution and the merge '
        'must still happen.',
        gen_fusion_order,
    ),
    'conv_wide': (
        'CONV2D with input channel counts around and above multiples of 16, integer and float, with and '
        'without RESCALE.',
        gen_conv_wide,
    ),
    'conv_shapes': (
        'CONV2D followed by RESCALE across resolutions, channel counts, strides and kernel sizes.',
        gen_conv_shapes,
    ),
    'conv_zero_points': (
        'CONV2D 3x3 with stride and dilation 1 over int8: batches, asymmetric and zero padding, tiny and odd '
        'outputs, channel counts around multiples of four, input and weight zero points, a merged RESCALE, '
        'and sums next to the int32 range: input -128 with zero point 127 and weights 127 with zero point '
        '-128 over 900 and 1024 channels.',
        gen_conv_zero_points,
    ),
    'tconv_taps': (
        'TRANSPOSE_CONV2D over an input and weights of ones, so that every output element counts the taps that '
        'reach it, across strides, kernel sizes and positive and negative out_pad.',
        gen_tconv_taps,
    ),
    'conv_tiles': (
        'int8 CONV2D beyond 3x3 stride 1: rectangular kernels, strides and dilations up to 3, padding wider '
        'than the kernel, batches, input and weight zero points, channel counts off multiples of 4, RESCALE '
        'fusion, and a layer whose tile does not fit into workgroup shared memory.',
        gen_conv_tiles,
    ),
    'conv_lanes': (
        'CONV3D, DEPTHWISE_CONV2D, TRANSPOSE_CONV2D and MATMUL in float32 and float16 with values whose products '
        'are inexact, so that a contracted or reassociated sum shows, and in int8 with channel counts that run '
        'the four-wide loop together with its tail.',
        gen_conv_lanes,
    ),
    'conv_float': (
        'CONV2D over float input and weights: f32 with an f32 accumulator and f16 with an f16 and with an f32 '
        'one, across channel counts, kernel sizes, strides, dilation, padding and batches. The values mix '
        'magnitudes, so that a kernel adding the products in another order than the direct one is caught, and '
        'the padded cases tell a position the direct kernel skips from one that adds a zero product.',
        gen_conv_float,
    ),
    'resize': (
        'RESIZE across both modes, upscaling and downscaling, fractional scales, negative offsets and '
        'borders, channel counts off multiples of 4, batches and every supported element type.',
        gen_resize,
    ),
    'concat': (
        'CONCAT along every axis, over two and three inputs, across ranks and element types, with sizes that '
        'are not multiples of four.',
        gen_concat,
    ),
    'table': (
        'TABLE across ranks, element types and last dimensions that are and are not multiples of four.',
        gen_table,
    ),
    'pool': (
        'MAX_POOL2D and AVG_POOL2D across channel counts on and off multiples of four, window sizes, strides, '
        'padding, batches and element types. The float graphs matter most: their inputs are random bits, so '
        'NaNs and infinities land in the windows, and in MAX_POOL2D the first NaN of a window decides the '
        'result of its channel.',
        gen_pool,
    ),
    'fft': (
        'FFT2D forward and inverse and RFFT2D over power-of-two shapes, batches and planes smaller and larger '
        'than a workgroup.',
        gen_fft,
    ),
    'bench': (
        'Large tensors per operator family on 32-bit and narrower elements, plus three model-like graphs: a '
        'quantized CNN block, a float attention block and a quantized per-channel affine transform with a '
        'lookup.',
        gen_bench,
    ),
    'bench_fusion': (
        'Accumulating operators followed by a RESCALE on large tensors.',
        gen_fusion_bench,
    ),
    'bench_conv_grid': (
        'Timing grid for the CONV2D kernels: 3x3 stride 1 int8 convolutions without RESCALE over resolutions '
        'and channel counts.',
        gen_conv_decomp,
    ),
    'bench_conv_intermediate': (
        'The key conv_decomp layers with an identity CLAMP in front, so that CONV2D reads a tensor allocated '
        "by the graph layer instead of the application's input tensor.",
        gen_conv_mid,
    ),
    'bench_narrow': (
        'Large int8 and int16 tensors for MUL, SELECT, ARITHMETIC_RIGHT_SHIFT, RESCALE, GATHER and SCATTER, '
        'whose 32-bit forms are bound by memory bandwidth.',
        gen_bench_narrow,
    ),
    'bench_ops': (
        'Large tensors for CONV3D, RESHAPE and the reductions along an axis that is not the last one.',
        gen_bench_ops,
    ),
}


def generate(out_dir, suites):
    total = 0
    for name in suites:
        description, generator = SUITES[name]
        GRAPHS.clear()
        generator()
        names = [g.name for g in GRAPHS]
        assert len(names) == len(set(names)), 'duplicate graph names in ' + name
        suite_dir = os.path.join(out_dir, name)
        os.makedirs(suite_dir, exist_ok=True)
        manifest = []
        for g in GRAPHS:
            with open(os.path.join(suite_dir, g.name + '.spvasm'), 'w', newline='\n') as handle:
                handle.write(render(g.m, [(e, d) for e, d, _ in g.inputs], g.input_ids, g.outputs))
            manifest.append(f'graph {g.name} {g.name}.spvasm')
            for e, d, seed in g.inputs:
                manifest.append(f'input {ELEM[e][3]} {",".join(map(str, d))} {seed}')
            for _, e, d in g.outputs:
                manifest.append(f'output {ELEM[e][3]} {",".join(map(str, d))}')
            manifest.append('end')
        with open(os.path.join(suite_dir, 'manifest.txt'), 'w', newline='\n') as handle:
            handle.write('\n'.join(manifest) + '\n')
        total += len(GRAPHS)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', help='directory that receives one folder per suite')
    parser.add_argument('--suites', help='comma separated suite names, every suite by default')
    parser.add_argument('--list', action='store_true', help='print the suites and exit')
    args = parser.parse_args()
    if args.list:
        for name, (description, _) in SUITES.items():
            print(f'{name}: {description}')
        return 0
    if not args.out:
        parser.error('--out is required')
    suites = args.suites.split(',') if args.suites else list(SUITES)
    unknown = [name for name in suites if name not in SUITES]
    if unknown:
        parser.error('unknown suites: ' + ', '.join(unknown))
    print(f'{generate(args.out, suites)} graphs in {len(suites)} suites written to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
