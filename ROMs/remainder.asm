LDA numerator
LDY divisor
JSR remainder
BRK


remainder:
    PHA ; A is numerator
    TYA
    PHA ; Y is divisor
    TSX
    LDA $102,X
    LDY #0
divloop:
    SEC
    SBC $101,X
    BEQ done
    BCC done
    INY
    JMP divloop
done:
    BEQ skipadd
    CLC
    ADC $101,X
skipadd:
    STA $102,X
    PLA
    PLA
    RTS

section .data
    numerator: .byte #52
    divisor: .byte #5