from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from typing import Dict, Callable, Tuple, Union
from enum import Enum


class AddressMode(Enum):
    Accumulator = 0
    Immediate = 1
    ZeroPage = 2
    ZeroPageX = 3
    ZeroPageY = 4
    Relative = 5
    Absolute = 6
    AbsoluteX = 7
    AbsoluteY = 8
    Indirect = 9
    IndirectX = 10
    IndirectY = 11


class Condition(Enum):
    CarryClear = 0
    CarrySet = 1
    ZeroSet = 2
    ZeroClear = 3
    NegativeSet = 4
    NegativeClear = 5
    OverflowSet = 6
    OverflowClear = 7


class ProcessStatusRegister:
    """Process status/flags registers"""

    def __init__(self: ProcessStatusRegister) -> None:
        self.carry: bool = False  # C
        self.zero: bool = False  # Z
        self.interrupt_disable: bool = False  # I
        self.decimal_mode: bool = False  # D
        self.break_command: bool = False  # B
        self.fifth_bit: bool = True  # 1
        self.overflow: bool = False  # V
        self.negative: bool = False  # N

        self.register = np.uint8(0)

    def compile_flags(self: ProcessStatusRegister) -> None:
        for bitpos, flag in enumerate(
            (
                self.carry,
                self.zero,
                self.interrupt_disable,
                self.decimal_mode,
                self.break_command,
                self.fifth_bit,
                self.overflow,
                self.negative,
            )
        ):
            self.register |= (flag & 0b1) << bitpos

    def expand_flags(self: ProcessStatusRegister) -> None:
        self.carry = bool(self.register & 0b1)
        self.zero = bool((self.register >> 1) & 0b1)
        self.interrupt_disable = bool((self.register >> 2) & 0b1)
        self.decimal_mode = bool((self.register >> 3) & 0b1)
        self.break_command = bool((self.register >> 4) & 0b1)
        self.fifth_bit = bool((self.register >> 5) & 0b1)
        self.overflow = bool((self.register >> 6) & 0b1)
        self.negative = bool((self.register >> 7) & 0b1)


class RAM:
    """RAM mapping
    $0000-$00FF -- Zero page
    $0100-$01FF -- Stack
    $0200-$FFFA -- FREE
    $FFFA,$FFFB -- non-maskable interrupt handler
    $FFFC,$FFFD -- power on reset location
    $FFFE,$FFFF -- BRK/interrupt handler
    """

    def __init__(self: RAM) -> None:
        self.ram: npt.NDArray[np.uint8] = np.ndarray(0)

    def __getitem__(self, key: np.uint16):
        return self.ram[key]

    def __setitem__(self, key: np.uint16, value: np.uint8):
        self.ram[key] = value

    def set_reset_vector(self: RAM, vector: np.uint8) -> None:
        self.ram[np.uint8(0xFFFC)] = np.uint8(vector & 0xFF)
        self.ram[np.uint8(0xFFFD)] = np.uint8((vector >> 16) & 0xFF)


class CPU:
    """CPU"""

    def __init__(self: CPU, ROM_path: str, verbose: bool) -> None:
        # Registers
        self.A = np.uint8(0)  # Accumulator
        self.X = np.uint8(0)  # rX
        self.Y = np.uint8(0)  # rY
        self.S = np.uint8(0)  # Stack pointer
        self.P = ProcessStatusRegister()  # Process Status / Flags
        self.IR = np.uint32(0)  # Instruction Register
        self.PC = np.uint16(0)  # Program Counter
        # RAM
        self.RAM = RAM()
        ROM = self._load_ROM(ROM_path)
        self.RAM.ram = np.concatenate(np.zeros(0x0200, dtype=np.uint8), ROM)
        self.RAM.ram = np.concatenate(
            self.RAM.ram, np.zeros(0x10000 - self.RAM.ram.size, dtype=np.uint8)
        )
        self.RAM.set_reset_vector(np.uint8(0x0200))
        # Get reset vector
        self.PC = self._get_reset_vector_0xFFFC_0xFFFD()
        # Opcode-to-function map
        # Value is opcode function and its arguments specifying which version of the opcode to use
        # e.g. ASL zpg,X --> self._asl, address_mode=ZeroPageX
        # This just makes it so we don't have to write out *every single op code possible*
        # We can condense some of them
        self.opcode_map: Dict[
            int, Tuple[Callable, Dict[str, Union[AddressMode, Condition]]]
        ] = {
            # 00-0F
            0x00: (self._brk, {}),
            # 0x01: ,
            # 0x05: ,
            0x06: (self._asl, {"address_mode": AddressMode.ZeroPage}),
            # 0x08: ,
            # 0x09: ,
            0x0A: (self._asl, {"address_mode": AddressMode.Accumulator}),
            # 0x0D: ,
            0x0E: (self._asl, {"address_mode": AddressMode.Absolute}),
            # # 10-1F
            0x10: (
                self._bxx,
                {
                    "address_mode": AddressMode.Relative,
                    "condition": Condition.NegativeClear,
                },
            ),
            # 0x11: ,
            # 0x15: ,
            0x16: (self._asl, {"address_mode": AddressMode.ZeroPageX}),
            0x18: (self._clc, {}),
            # 0x19: ,
            # 0x1D: ,
            0x1E: (self._asl, {"address_mode": AddressMode.AbsoluteX}),
            # # 20-2F
            # 0x20: ,
            # 0x21: ,
            # 0x24: ,
            # 0x25: ,
            # 0x26: ,
            # 0x28: ,
            # 0x29: ,
            # 0x2A: ,
            # 0x2C: ,
            # 0x2D: ,
            # 0x2E: ,
            # # 30-3F
            # 0x30: ,
            # 0x31: ,
            # 0x35: ,
            # 0x36: ,
            # 0x38: ,
            # 0x39: ,
            # 0x3D: ,
            # 0x3E: ,
            # # 40-4F
            # 0x40: ,
            # 0x41: ,
            # 0x45: ,
            # 0x46: ,
            # 0x48: ,
            # 0x49: ,
            # 0x4A: ,
            # 0x4C: ,
            # 0x4D: ,
            # 0x4E: ,
            # # 50-5F
            # 0x50: ,
            # 0x51: ,
            # 0x55: ,
            # 0x56: ,
            # 0x58: ,
            # 0x59: ,
            # 0x5D: ,
            # 0x5E: ,
            # # 60-6F
            # 0x60: ,
            # 0x61: ,
            # 0x65: ,
            # 0x66: ,
            # 0x68: ,
            # 0x69: ,
            # 0x6A: ,
            # 0x6C: ,
            # 0x6D: ,
            # 0x6E: ,
            # # 70-7F
            # 0x70: ,
            # 0x71: ,
            # 0x75: ,
            # 0x76: ,
            # 0x78: ,
            # 0x79: ,
            # 0x7D: ,
            # 0x7E: ,
            # # 80-8F
            # 0x81: ,
            # 0x84: ,
            # 0x85: ,
            # 0x86: ,
            # 0x88: ,
            # 0x8A: ,
            # 0x8C: ,
            # 0x8D: ,
            # 0x8E: ,
            # # 90-9F
            # 0x90: ,
            # 0x91: ,
            # 0x94: ,
            # 0x95: ,
            # 0x96: ,
            # 0x98: ,
            # 0x99: ,
            # 0x9A: ,
            # 0x9D: ,
            # # A0-AF
            # 0xA0: ,
            # 0xA1: ,
            # 0xA2: ,
            # 0xA4: ,
            # 0xA5: ,
            # 0xA6: ,
            # 0xA8: ,
            # 0xA9: ,
            # 0xAA: ,
            # 0xAC: ,
            # 0xAD: ,
            # 0xAE: ,
            # # B0-BF
            # 0xB0: ,
            # 0xB1: ,
            # 0xB4: ,
            # 0xB5: ,
            # 0xB6: ,
            # 0xB8: ,
            # 0xB9: ,
            # 0xBA: ,
            # 0xBC: ,
            # 0xBD: ,
            # 0xBE: ,
            # # C0-CF
            # 0xC0: ,
            # 0xC1: ,
            # 0xC4: ,
            # 0xC5: ,
            # 0xC6: ,
            # 0xC8: ,
            # 0xC9: ,
            # 0xCA: ,
            # 0xCC: ,
            # 0xCD: ,
            # 0xCE: ,
            # # D0-DF
            # 0xD0: ,
            # 0xD1: ,
            # 0xD5: ,
            # 0xD6: ,
            # 0xD8: ,
            # 0xD9: ,
            # 0xDD: ,
            # 0xDE: ,
            # # E0-EF
            # 0xE0: ,
            # 0xE1: ,
            # 0xE4: ,
            # 0xE5: ,
            # 0xE6: ,
            # 0xE8: ,
            # 0xE9: ,
            # 0xEA: ,
            # 0xEC: ,
            # 0xED: ,
            # 0xEE: ,
            # # F0-FF
            # 0xF0: ,
            # 0xF1: ,
            # 0xF5: ,
            # 0xF6: ,
            # 0xF8: ,
            # 0xF9: ,
            # 0xFD: ,
            # 0xFE: ,
        }
        self._verbose = verbose

    def _load_ROM(self: CPU, ROM_path: str) -> npt.NDArray[np.uint8]:
        with open(ROM_path, "rb") as rombytefile:
            file_bytes = rombytefile.read()
            return np.frombuffer(file_bytes, dtype=np.uint8)

    def _get_reset_vector_0xFFFC_0xFFFD(self: CPU) -> np.uint16:
        return np.uint16(
            np.uint16(self.RAM[np.uint16(0xFFFD)] << 8)
            | np.uint16(self.RAM[np.uint16(0xFFFC)])
        )

    def _cycle(self: CPU, count: int) -> None:
        for _ in range(count):
            time.sleep(1 * 10 ^ -6)

    def _load_next_opcode(self: CPU, addr: np.uint16) -> np.uint32:
        bytes = [self.RAM[np.uint16(self.PC + ii)] for ii in range(4)]
        return np.uint32(
            (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]
        )

    def fetch_decode_execute(self: CPU) -> None:
        self.IR = self._load_next_opcode(self.PC)
        opcode: int = int((self.IR >> 24) & 0xFF)  # 8-bit int
        opfunc, opargs = self.opcode_map[opcode]
        opfunc(**opargs)

    def main_loop(self: CPU) -> None:
        if self._verbose:
            print("| INST |             |")
            print("| ADDR | AR XR YR SP |")
        while True:
            if self._verbose:
                self.examine()
            self.fetch_decode_execute()

    def _get_args(self: CPU) -> Tuple[np.uint8, np.uint16]:
        arg1: np.uint8 = np.uint8((self.IR >> 16) & 0xFF)
        arg2: np.uint8 = np.uint8((self.IR >> 8) & 0xFF)
        eightbitarg = arg2
        sixteenbitarg = np.uint16(np.uint16(arg2) << 8 | np.uint16(arg1))
        return eightbitarg, sixteenbitarg

    # Opcode functions

    def _adc(self: CPU, address_mode: AddressMode) -> None:
        """Add with carry: A+M+C"""
        _8bitarg, _16bitarg = self._get_args()
        oldA = self.A
        cycles = 0
        opcode_bytes = 1
        match address_mode:
            case AddressMode.Immediate:
                self.A += np.uint8(_8bitarg)
                cycles = 2
                opcode_bytes = 2
            case AddressMode.ZeroPage:
                self.A += self.RAM[np.uint16(_8bitarg)]
                cycles = 3
                opcode_bytes = 2
            case AddressMode.ZeroPageX:
                self.A += self.RAM[np.uint16(_8bitarg + self.X)]
                cycles = 4
                opcode_bytes = 2
            case AddressMode.ZeroPageY:
                self.A += self.RAM[np.uint16(_8bitarg + self.Y)]
                cycles = 4
                opcode_bytes = 2
            case AddressMode.Absolute:
                self.A += self.RAM[_16bitarg]
                cycles = 4
                opcode_bytes = 3
            case AddressMode.AbsoluteX:
                self.A += self.RAM[np.uint16(_16bitarg + self.X)]
                cycles = 4
                opcode_bytes = 3
            case AddressMode.AbsoluteY:
                self.A += self.RAM[np.uint16(_16bitarg + self.Y)]
                cycles = 4
                opcode_bytes = 3
            # NOTE: Fix later, these ones are weird
            # case AddressMode.IndirectX:
            #     self.A += self.RAM[self.RAM[eightbitarg + self.X]]
            # case AddressMode.IndrectY:
            #     self.A += self.RAM[self.RAM]
        if oldA > self.A:
            self.P.carry = True
        if self.A == 0:
            self.P.zero = True
        if (self.A >> 7) & 0b1 == 1:
            self.P.negative = True
        # NOTE: Figure out how to write case for setting overflow
        self._cycle(cycles)
        self.PC += np.uint16(opcode_bytes)

    def _and(self: CPU, address_mode: AddressMode) -> None:
        """logical AND: A&M"""
        _8bitarg, _16bitarg = self._get_args()
        cycles = 0
        opcode_bytes = 1
        match address_mode:
            case AddressMode.Immediate:
                cycles = 2
                self.A &= _8bitarg
                opcode_bytes = 2
            case AddressMode.ZeroPage:
                cycles = 3
                self.A &= self.RAM[np.uint16(_8bitarg)]
                opcode_bytes = 2
            case AddressMode.ZeroPageX:
                cycles = 4
                self.A &= self.RAM[np.uint16(_8bitarg + self.X)]
                opcode_bytes = 2
            case AddressMode.ZeroPageY:
                cycles = 4
                self.A &= self.RAM[np.uint16(_8bitarg + self.Y)]
                opcode_bytes = 2
            case AddressMode.Absolute:
                cycles = 4
                self.A &= self.RAM[_16bitarg]
                opcode_bytes = 3
            case AddressMode.AbsoluteX:
                cycles = 4
                self.A &= self.RAM[np.uint16(_16bitarg + self.X)]
                opcode_bytes = 3
            case AddressMode.AbsoluteY:
                cycles = 4
                self.A &= self.RAM[np.uint16(_16bitarg + self.Y)]
                opcode_bytes = 3
            # NOTE: Fix later, these ones are weird
            # case AddressMode.IndirectX:
            #     self.A += self.RAM[self.RAM[eightbitarg + self.X]]
            # case AddressMode.IndrectY:
            #     self.A += self.RAM[self.RAM]
        if self.A == 0:
            self.P.zero = True
        if (self.A >> 7) & 0b1 == 1:
            self.P.negative = True
        self._cycle(cycles)
        self.PC += np.uint16(opcode_bytes)

    def _asl(self: CPU, address_mode: AddressMode) -> None:
        """Arithmetic shift left: A*2 or M*2"""
        _8bitarg, _16bitarg = self._get_args()
        cycles = 0
        opcode_bytes = 1
        match address_mode:
            case AddressMode.Accumulator:
                self.P.carry = bool((self.A >> 7) & 0b1)
                self.A <<= 1
                cycles = 2
                opcode_bytes = 1
            case AddressMode.ZeroPage:
                self.P.carry = bool((self.RAM[np.uint16(_8bitarg)] >> 7) & 0b1)
                self.RAM[np.uint16(_8bitarg)] <<= 1
                cycles = 5
                opcode_bytes = 2
            case AddressMode.ZeroPageX:
                self.P.carry = bool((self.RAM[np.uint16(_8bitarg + self.X)] >> 7) & 0b1)
                self.RAM[np.uint16(_8bitarg + self.X)] <<= 1
                cycles = 6
                opcode_bytes = 2
            case AddressMode.Absolute:
                self.P.carry = bool((self.RAM[_16bitarg] >> 7) & 0b1)
                self.RAM[_16bitarg] <<= 1
                cycles = 6
                opcode_bytes = 3
            case AddressMode.AbsoluteX:
                self.P.carry = bool(
                    (self.RAM[np.uint16(_16bitarg + self.X)] >> 7) & 0b1
                )
                self.RAM[np.uint16(_16bitarg + self.X)] <<= 1
                cycles = 7
                opcode_bytes = 3
        if self.A == 0:
            self.P.zero = True
        if (self.A >> 7) & 0b1 == 1:
            self.P.negative = True
        self._cycle(cycles)
        self.PC += np.uint16(opcode_bytes)

    def _bxx(self: CPU, address_mode: AddressMode, condition: Condition) -> None:
        """Branch (relative) if ..."""
        _8bitarg, _ = self._get_args()
        cycles = 0
        opcode_bytes = 2
        match address_mode:
            case AddressMode.Relative:
                match condition:
                    case Condition.CarryClear:  # BCC
                        if not self.P.carry:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.CarrySet:  # BCS
                        if self.P.carry:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.ZeroSet:  # BEQ
                        if self.P.zero:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.NegativeSet:  # BMI
                        if self.P.negative:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.ZeroClear:  # BNE
                        if not self.P.zero:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.NegativeClear:  # BPL
                        if not self.P.negative:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.OverflowClear:  # BVC
                        if not self.P.overflow:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
                    case Condition.OverflowSet:  # BVS
                        if self.P.overflow:
                            self.PC += np.uint16(_8bitarg)
                            cycles = 3
                            opcode_bytes = 0
                        else:
                            cycles = 2
        self._cycle(cycles)
        self.PC += np.uint16(opcode_bytes)

    def _bit(self: CPU, address_mode: AddressMode) -> None:
        """Test bit"""
        _8bitarg, _16bitarg = self._get_args()
        cycles = 0
        opcode_bytes = 1
        match address_mode:
            case AddressMode.ZeroPage:
                if (self.A & self.RAM[np.uint16(_8bitarg)]) == 0:
                    self.P.zero = True
                self.P.overflow = bool((self.RAM[np.uint16(_8bitarg)] >> 6) & 0b1)
                self.P.negative = bool((self.RAM[np.uint16(_8bitarg)] >> 7) & 0b1)
                opcode_bytes = 2
                cycles = 3
            case AddressMode.Absolute:
                if (self.A & self.RAM[_16bitarg]) == 0:
                    self.P.zero = True
                self.P.overflow = bool((self.RAM[_16bitarg] >> 6) & 0b1)
                self.P.negative = bool((self.RAM[_16bitarg] >> 7) & 0b1)
                opcode_bytes = 3
                cycles = 4
        self._cycle(cycles)
        self.PC += np.uint16(opcode_bytes)

    def _brk(self: CPU) -> None:
        self.P.break_command = True
        self.PC = np.uint16(
            np.uint16(self.RAM[np.uint16(0xFFFF)] << 8)
            | np.uint16(self.RAM[np.uint16(0xFFFE)])
        )
        self._cycle(7)

    def _clc(self: CPU) -> None:
        """Clear Carry"""
        self.P.carry = False
        self._cycle(2)
        self.PC += 1

    def _cld(self: CPU) -> None:
        """Clear Decimal Mode"""
        self.P.decimal_mode = False
        self._cycle(2)
        self.PC += 1

    def _cli(self: CPU) -> None:
        """Clear interrupt disable"""
        self.P.interrupt_disable = False
        self._cycle(2)
        self.PC += 1

    def _clv(self: CPU) -> None:
        """Clear overflow flag"""
        self.P.overflow = False
        self._cycle(2)
        self.PC += 1

    def _cmp(self: CPU, address_mode=AddressMode) -> None:
        """Test bit"""
        _8bitarg, _16bitarg = self._get_args()
        cycles = 0
        opcode_bytes = 1
        match address_mode:
            case AddressMode.Immediate:
                self.P.carry = bool(self.A > _8bitarg)
                self.P.zero = bool(self.A - _8bitarg == 0)
                self.P.negative = bool((np.uint8(self.A - _8bitarg) >> 7) & 0b1)
                cycles = 2
                opcode_bytes = 2

    def examine(self: CPU) -> None:
        pc = self.PC
        ar = self.A
        xr = self.X
        yr = self.Y
        sp = self.S
        print(f"| {pc:04} | {ar:02} {xr:02} {yr:02} {sp:02} |")
