import random

l = 42

print("NOTE: Only ever copy the bits or the bases, not both - they are linked.")
s = [random.choice([0, 1]) for _ in range(l)]
print("".join([str(c) for c in s]))
s1 = ["+" if s[i] == 0 else "x" for i in range(l)]
print("".join(s1))


