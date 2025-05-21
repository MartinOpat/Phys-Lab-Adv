s1 = "++++x+x+++x+x+x+x+x+++++++xx+xx+x+xx++xx+x"
s2 = "+xxx++x+x+x+xx+x+xxxx+x+x++++xxx+++x+xxx++"
bs = "11110000110010010101100001110100000010110"

key = []
for i in range(len(s1)):
    if s1[i] == s2[i]:
        key.append(bs[i])

print("".join(key))
