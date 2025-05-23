# -*- coding: utf-8 -*-
"""RSA_protocol_student_worksheet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13dS4YHoFnPZ4JH3AtQUd3BsYH630VIEq

First import the relevant libraries
"""

# Python for RSA asymmetric cryptographic algorithm.
# For demonstration, values are
# relatively small compared to practical application
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import randprime # This to generate random primes

from numba import njit  # This is to be fast(er)

# Set random seed(s) for replicability
np.random.seed(0)

"""The gcd is defined as:"""

# From the template
def gcd(a, h):
    temp = 0
    while(1):
        temp = a % h
        if (temp == 0):
            return h
        a = h
        h = temp

# Own version using Euclid
def gcd_euclid(a, h):
    while h != 0:
        h, a = a % h, h  # NOTE: This has to stay in the same line
    return a

# But why not just use the one from the standard library?
gcd_std = math.gcd

# Or even better the numpy version for c-level speed
gcd_fast = np.gcd

"""Now choose p, q, n, e and phi."""

# Use built in function to generate random primes
# We only need to execute randprime once to get random values
# Afterwards, we shall re-use these same values for replicability

# p = randprime(10, 100)
# q = randprime(10, 100)
# # Ensure p and q are not the same not to break the RSA protocol
# while p == q:
#     q = randprime(10, 100)

p = np.int64(67)
q = np.int64(97)


print(f"First prime: {p}")
print(f"Second prime: {q}")

n = p * q
phi = (p-1) * (q-1)
print(f"Composed number: n = {p}*{q} = {n}")
print(f"Phi: {phi}")


# Find suitable e
@njit
def find_e(phi: np.int64):
    e = np.int64(2)
    while (e < phi):
        # e must be co-prime to phi and
        # smaller than phi.
        if(gcd_fast(e, phi) == 1):
            break
        else:
            e = e + 1
    return e

e = find_e(phi)
print(f"e = {e}")

"""This is the code to find the lcm."""

# Python Program to find the L.C.M. of two input number

def compute_lcm(x, y):

   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1

   return lcm

# Own lcm using gcd
def compute_lcm_own(x, y):
    return x*y // gcd(x, y)

# Fast numpy version
compute_lcm_fast = np.lcm

"""Now compute the lcm for your variables"""

lambda_n = compute_lcm_fast(p-1, q-1)
print(f"lambda(n) = {lambda_n}")

# Fast modular exponentiation function with numpy support
@njit
def pow_mod(b: np.int64, e: np.int64, m: np.int64):
    res = np.int64(1)
    b = b % m
    while e > 0:
        if e & 1:
            res = (res * b) % m
        b = (b * b) % m
        e >>= 1
    return res

"""Define d"""

d = pow_mod(e, np.int64(-1), lambda_n)
print(f"d = {d}")

"""Choose a message"""

M = np.int64(21)
print(f"Message: {M}")

"""Now encrypt the message"""

C = pow_mod(M, e, n)
print(f"Encrypted message: {C}")

# A fast function to calculate the d-th root of a number
# This function assumes the result is an integer
# Otherwise, it returns the smallest integer i s.t.: i^d < val
@njit
def root_d(val: np.int64, d: np.int64):
    ans = 1
    while ans**d < val:
        ans += 1
    return ans if ans**d == val else ans - 1

"""Because you know all variables you can now decrypt the message"""

Cd = M % n
C = root_d(Cd, d)

print(f"Decrypted message: {C}")

"""Now you should see that your RSA protecol indeed functions

**But what if you are not in the communication loop, but would like to eveasedrop?**

If you have intercepted an encrypted message, you can crack it. Assume you have found the encrypted message from above, now the goal is to break the encryption

First state the variables you have
"""

# We know the encrypted message, e and n

"""Find p and q."""

#We write a function that performs prime factorization in order to extract p and q from n
#this is the compuationally heavay part of the process but it's easy with the small primes

@njit
def prime_factors(n):
    factors = []

    # Add all factors of 2
    while not n & 1:
        factors.append(2)
        n //= 2

    # Add odd factors from 3 up to sqrt(n)
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2

    # If n > 2, it must be prime
    if n > 2:
        factors.append(n)

    return np.array(factors)

factors = prime_factors(n)
p_cracked, q_cracked = factors
print(f"The cracked prime factors of n={n} are: {factors}")

"""Now find the lcm."""

# Since we know the primes now, we use the same formula from before
lambda_n_cracked = compute_lcm_fast(p_cracked-1, q_cracked-1)
print(f"lambda(n) [cracked] = {lambda_n_cracked}")

"""Now you can find d."""

d_cracked = pow_mod(e, np.int64(-1), lambda_n_cracked)
print(f"Cracked private key: {d_cracked}")

"""Using d, decrypt the message."""

Cd_cracked = M % n
C_cracked = root_d(Cd, d_cracked)
print(f"Decrypted message: {C_cracked}")

# This code is written by Marianne Westerhof
# This code is contributed by Pranay Arora.