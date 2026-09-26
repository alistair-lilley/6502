JSR bubblesort
;JSR examine_list
BRK


bubblesort:

outerloop:
; set swapped to false
    LDA #0
    STA swapped
    LDA #0
    STA innercounter

innerloop:
; compare a[j] and a[j+1]
; if
    LDX innercounter
    LDY innercounter
    INY
    LDA numbers,X
    CMP numbers,Y
    BCC endif ; jump if a[j] <= a[j+1]

; a[j] > a[j+1]
    STA swap
    LDA numbers,Y
    STA numbers,X
    LDA swap
    STA numbers,Y
    LDA #1
    STA swapped

endif:
    INC innercounter
    LDA len
; innercounter == len - outercounter - 1?
    SEC
    SBC outercounter
    SEC
    SBC #1
    CMP innercounter
    BNE innerloop

    LDA swapped
    BEQ done
    INC outercounter
    LDA len
    CMP outercounter
    BNE outerloop
done:
    RTS




;examine_list:
;    LDA #0
;    NOP
;    NOP
;    NOP
;    LDY len
;
;load_backwards:
;    DEY
;    LDA numbers,Y
;    PHA
;    CPY #0
;    BNE load_backwards
;    
;    LDY len
;load_in_order:
;    PLA
;    DEY
;    BNE load_in_order
;
;    RTS





section .data
    numbers: .byte #$1c, #$9, #$3f, #$58, #$04, #$5a, #$28, #$4c, #$3, #$23, #$3c, #$60, #$6f, #$41, #$1b
;    numbers: .byte #$1c, #$9, #$3f, #$58, #$04, #$5a, #$28
;    numbers: .byte #05, #01, #03

    len: .byte #$F

    outercounter: .byte #$0
    innercounter: .byte #$0

    swap: .byte #$0
    swapped: .byte #$0