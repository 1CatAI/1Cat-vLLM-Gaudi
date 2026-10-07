# SPDX-License-Identifier: Apache-2.0
import pytest
from tools.audit_deepseek_v41_tpc_loops import empty_self_loops


def test_actual_hoisted_poll_with_aliased_loop_label_is_rejected():
    assembly = '''TPC architecture: gaudi2
 1180: ff 2d ad 00 ld_l mmio S11, 0xd74; nop; nop; nop
 11e0: 90 7f 00 00 nop; jmpr .LBB0_64, !SP6; nop; nop
.LBB0_64:
00008940 .LBB0_65:
 8940: 90 7f 00 00 nop; jmpr .LBB0_64; nop; nop
'''
    assert empty_self_loops(assembly)[0]['start'] == '0x8940'


def test_poll_reloading_inside_loop_and_bounded_counter_are_not_empty():
    assembly = '''TPC architecture: gaudi2
.LBB0_1:
 1100: 90 7f 00 00 ld_l mmio S11, 0xd74; nop; nop; nop
 1120: 90 7f 00 00 nop; jmpr .LBB0_1; nop; nop
.LBB0_2:
 1140: 90 7f 00 00 nop; add.i32 S1, S1, 1; nop; nop
 1160: 90 7f 00 00 nop; jmpr .LBB0_2, SP1; nop; nop
'''
    assert not empty_self_loops(assembly)


def test_wrong_disassembly_architecture_is_rejected():
    with pytest.raises(ValueError, match='Gaudi2'):
        empty_self_loops('file format elf32-i386\n 10: eb fe jmp 0x10')
