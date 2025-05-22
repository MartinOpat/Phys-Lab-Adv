# Letters
words = open("w.txt", "r").read().split("\n")[:-1]
res = open("results3.txt", "r").read().split("\n")[:-1]
resEve = open("results3-eve.txt", "r").read().split("\n")[:-1]


# letter error
err = 0
errEve = 0

for i, w in enumerate(words):
    for j, c in enumerate(w):
        if c != res[i][j]:
            err += 1
        if c != resEve[i][j]:
            errEve += 1

print("Error in letters without Eve:", err)
print("Error in leters with Eve", errEve)

# Bits
bitsSent = open("results1.txt", "r").read().split("\n")[:-1]
bitsRec = open("results2.txt", "r").read().split("\n")[:-1]

bitsSentEve = open("results1-eve.txt", "r").read().split("\n")[:-1]
bitsRecEve = open("results2-eve.txt", "r").read().split("\n")[:-1]

berr = 0
berrEve = 0

for i, bs in enumerate(bitsSent):
    for j, b in enumerate(bs):
        if b != bitsRec[i][j]:
            berr += 1


for i, bs in enumerate(bitsSentEve):
    for j, b in enumerate(bs):
        if b != bitsRecEve[i][j]:
            berrEve += 1


print("Bit errors without Eve:", berr)
print("Bit errors with Eve:", berrEve)


