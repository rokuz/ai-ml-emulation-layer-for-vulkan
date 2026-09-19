#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Checks a build of the emulation layers against a reference, graph by graph and byte by byte.

The reference is either another build of the layers (--reference, typically the commit a change is based on) or the
build under test itself with every optional code path switched off (--self-check: no specialization, no operator
fusion, no convolution tiles, no integer dot product).

Two tables describe where a difference from an older build is expected. Outputs listed in TOLERANCES are compared
numerically, because their kernels no longer sum in the same order. Graphs listed in CORRECTED are not compared at
all, because the older build answers them wrongly; the tconv_taps suite pins the right answer for those, and every
graph for which opsuite.py knows the result analytically is checked against it as well."""
import argparse
import array
import os
import struct
import subprocess
import sys

import opsuite

SWITCHES = ['VMEL_DISABLE_SPECIALIZATION', 'VMEL_DISABLE_OPERATOR_FUSION', 'VMEL_DISABLE_CONV_TILES', 'VMEL_DISABLE_CONV_DOT']
TOLERANCES = {'fft2d_f32_': 1.0e-6}
CORRECTED = {'lanes_tconv2d_f16_2x3x5x5_oc6_k3_s13', 'lanes_tconv2d_f32_2x3x5x5_oc6_k3_s13',
             'lanes_tconv2d_i8_2x3x5x5_oc6_k3_s13', 'taps_tconv2d_i8_1x3x3x3_oc2_k4_s23_opm1111',
             'taps_tconv2d_i8_1x3x5x2_oc3_k3_s13_op0110', 'taps_tconv2d_i8_2x2x3x1_oc2_k2_s31_op2012'}


def run(runner, manifest, golden, layer, switches):
    env = dict(os.environ)
    for name in SWITCHES:
        env.pop(name, None)
        if switches:
            env[name] = '1'
    command = [runner, manifest, '--golden', golden]
    if layer:
        command += ['--layer', layer]
    result = subprocess.run(command, env=env, capture_output=True, text=True, errors='replace')
    if 'PASSED' not in result.stdout and 'FAILED' not in result.stdout:
        raise RuntimeError(f'{runner} did not finish on {manifest}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')


def tolerance(name):
    for prefix, value in TOLERANCES.items():
        if name.startswith(prefix):
            return value
    return None


def close(expected, actual, relative):
    if len(expected) != len(actual) or expected[:1] != actual[:1]:
        return False
    a, b = array.array('f'), array.array('f')
    a.frombytes(expected[1:])
    b.frombytes(actual[1:])
    scale = max((abs(v) for v in a), default=0.0)
    return all(abs(x - y) <= relative * scale for x, y in zip(a, b))


def checkReferences(actual_dir):
    failed = []
    for name, expected in opsuite.REFERENCES.items():
        path = os.path.join(actual_dir, name + '.bin')
        if not os.path.exists(path):
            continue
        data = open(path, 'rb').read()
        actual = struct.unpack(f'<{(len(data) - 1) // 4}i', data[1:]) if len(data) > 1 else ()
        wrong = sum(1 for a, b in zip(expected, actual) if a != b) + abs(len(expected) - len(actual))
        if wrong != 0:
            failed.append(f'{name}: {wrong} of {len(expected)} values differ from the computed result')
    return failed


def compare(expected_dir, actual_dir, tolerant):
    identical, tolerated, failed = 0, 0, []
    for name in sorted(os.listdir(expected_dir)):
        expected = open(os.path.join(expected_dir, name), 'rb').read()
        path = os.path.join(actual_dir, name)
        actual = open(path, 'rb').read() if os.path.exists(path) else b''
        if expected == actual:
            identical += 1
        elif tolerant and (name[:-4] in CORRECTED or
                           (tolerance(name) is not None and close(expected, actual, tolerance(name)))):
            tolerated += 1
        else:
            differing = sum(1 for x, y in zip(expected, actual) if x != y) + abs(len(expected) - len(actual))
            failed.append(f'{name[:-4]}: {differing} of {len(expected)} bytes differ')
    return identical, tolerated, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runner', required=True, help='path to the mlel_opsuite executable')
    parser.add_argument('--out', required=True, help='working directory for the graphs and the outputs')
    parser.add_argument('--layer', help='directory of the layers under test, VK_LAYER_PATH of the environment by default')
    parser.add_argument('--reference', help='directory of the reference layers')
    parser.add_argument('--self-check', action='store_true', help='use the layers under test without their optional paths')
    parser.add_argument('--suites', help='comma separated suite names, every suite by default')
    args = parser.parse_args()
    if bool(args.reference) == args.self_check:
        parser.error('exactly one of --reference and --self-check is required')

    suites = args.suites.split(',') if args.suites else list(opsuite.SUITES)
    graphs_dir = os.path.join(args.out, 'graphs')
    opsuite.generate(graphs_dir, suites)

    print(f'{"suite":26s} {"graphs":>6s} {"identical":>9s} {"tolerated":>9s} {"failed":>6s}')
    failures = []
    for suite in suites:
        manifest = os.path.join(graphs_dir, suite, 'manifest.txt')
        expected = os.path.join(args.out, 'expected', suite)
        actual = os.path.join(args.out, 'actual', suite)
        run(args.runner, manifest, expected, args.reference or args.layer, args.self_check)
        run(args.runner, manifest, actual, args.layer, False)
        identical, tolerated, failed = compare(expected, actual, not args.self_check)
        graphs = identical + tolerated + len(failed)
        failed += checkReferences(actual)
        print(f'{suite:26s} {graphs:6d} {identical:9d} {tolerated:9d} {len(failed):6d}', flush=True)
        failures += [f'{suite}/{line}' for line in failed]

    for line in failures:
        print('FAIL ' + line)
    print('FAILED' if failures else 'PASSED')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
