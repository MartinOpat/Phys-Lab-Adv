
from string import ascii_lowercase
import random

num = 30
words = set()
with open("w.txt", "w") as f:
    c = 0
    while c < num:
        c1 = random.choice(ascii_lowercase)
        c2 = random.choice(ascii_lowercase)
        c3 = random.choice(ascii_lowercase)
        word = c1+c2+c3

        if not word in words:
            f.write(word+"\n")
            c+=1
            words.add(word)
    
