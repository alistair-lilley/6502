LDA numerator
LDY divisor
JSR quotient
BRK


quotient:
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
    TYA
    STA $102,X
    PLA
    PLA
    RTS

section .data
    numerator: .byte #52
    divisor: .byte #5