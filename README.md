# READ ME

This project is a python implementation of a 6502 chip simulator, along with a few basic programs to run in it.

## Architecture

### Registers

The 6502 CPU has six 8-bit registers and one 16-bit register.

#### 8-bit registers

- I(nstruction)R(egister)
- A(ccumulator)
- X (memory addressing, counters)
- Y (memory addressing, counters)
- S(tack pointer)
- P(rocess status/flags)

#### Process status/flags register

- C - Carry Flag
- Z - Zero Flag
- I - Interrupt Disable
- D - Decimal Mode Flag
- B - Break Command
- V - Overflow Flag
- N - Negative Flag

#### 16-bit register

- P(rogram)C(ounter)

### RAM

It has 16-bit addressing space ($0000-$FFFF).

#### Memory Map

$0000-$00FF -- "Zero page" special addressing modes
$0100-$01FF -- System Stack
$0200-$FFFA -- General purpose memory
$FFFA/B -- non-maskable interrupt handler
$FFFC/D -- power on reset location
$FFFE/F -- BRK/interrupt handler