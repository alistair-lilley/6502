LDA num1
LDY num2
JSR multiply
BRK

multiply:
    PHA ; $104 -- return result high byte
    PHA ; $103 -- return result low byte
    PHA ; $102 -- num1
    TYA
    PHA ; $101 -- num2
    TSX
    LDA #0
    STA $104,X ; zero return result
    STA $103,X ; zero return result
    LDY #8
loop
    LSR $101,X
    BCC loop2
    CLC
    ADC $102,X
loop2
    ROR A
    ROR $104,X
    DEY
    BNE loop

    STA $103,X
    PLA ; pop num2
    PLA ; pop num1
    PLA ; pop low byte
    TAY
    PLA ; pop high byte
    RTS


section .data
    num1: .byte #$80
    num2: .byte #$02