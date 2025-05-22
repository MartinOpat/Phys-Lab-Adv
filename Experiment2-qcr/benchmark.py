#!/bin/python3

# Code to benchmark the RSA algorithm vs its breaking

import math
import matplotlib.pyplot as plt
import numpy as np
from sympy import randprime # This to generate random primes
from numba import njit, prange  # This is to be fast(er)
import time
import sys
from concurrent.futures import ThreadPoolExecutor  # Multithreading


gcd_fast = np.gcd
compute_lcm_fast = np.lcm

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

@njit
def mul_mod(a: np.int64, b: np.int64, m: np.int64) -> np.int64:
    """
    (a * b) % m  using the Russian-peasant method.
    Keeps every intermediate value < 2*m, so it never overflows int64
    as long as m < 2^62.
    """
    a %= m
    b %= m
    res = np.int64(0)
    while b:
        if b & 1:
            res = (res + a) % m
        a = (a + a) % m
        b >>= 1
    return res

@njit
def pow_mod(b: np.int64, e: np.int64, m: np.int64) -> np.int64:
    res = np.int64(1)
    b   = b % m
    while e:
        if e & 1:
            res = mul_mod(res, b, m)
        b = mul_mod(b, b, m)
        e >>= 1
    return res


@njit
def modinv(a: np.int64, m: np.int64):
    """Modular inverse using the extended Euclidean algorithm."""
    m0, x0, x1 = m, np.int64(0), np.int64(1)
    while a > 1:
        q = a // m
        a, m = m, a % m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

@njit
def prime_factors(n: np.int64):
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

@njit
def run_encrypt(primes: np.ndarray[np.int64]):
    ## Choose primes
    pidx = np.random.randint(0, len(primes))
    qidx = np.random.randint(0, len(primes))
    while qidx == pidx:
        qidx = np.random.randint(0, len(primes))
    p = primes[pidx]
    q = primes[qidx]

    # Find public key + info
    n = p * q
    phi = (p-1) * (q-1)
    e = find_e(phi)

    # Find private key
    lambda_n = compute_lcm_fast(p-1, q-1)
    d = modinv(e, lambda_n)

    # Generate and encrypt message
    M = np.random.randint(1, n-1)
    C = pow_mod(M, e, n)

    return M, C, n, d, e

@njit
def run_rsa(C: np.int64, d: np.int64, n: np.int64):
    # Decrypt message
    M = pow_mod(C, d, n)
    return M

@njit
def run_rsa_break(C: np.int64, n: np.int64, e: np.int64):
    # Extract prime factors
    factors = prime_factors(n)
    p_cracked, q_cracked = factors

    # Find private key
    lambda_n_cracked = compute_lcm_fast(p_cracked-1, q_cracked-1)
    d_cracked = modinv(e, lambda_n_cracked)

    # Decrypt message
    M = pow_mod(C, d_cracked, n)
    return M


def benchmark_rsa(primes, num_trials):
    encrypt_times = np.zeros(num_trials)
    decrypt_times = np.zeros(num_trials)
    break_times = np.zeros(num_trials)

    def experiment(_):
        start = time.perf_counter()
        M, C, n, d, e = run_encrypt(primes)
        t_encrypt = time.perf_counter() - start

        start = time.perf_counter()
        M1 = run_rsa(C, d, n)
        t_decrypt = time.perf_counter() - start

        start = time.perf_counter()
        M2 = run_rsa_break(C, n, e)
        t_break = time.perf_counter() - start

        if M1 != M:
            print(f"Decryption failed: {M1} != {M}")
        if M2 != M:
            print(f"Breaking failed: {M2} != {M}")

        return t_encrypt, t_decrypt, t_break

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(experiment, range(num_trials)))

    for i, (te, td, tb) in enumerate(results):
        encrypt_times[i] = te
        decrypt_times[i] = td
        break_times[i] = tb

    return encrypt_times, decrypt_times, break_times


# Helper function to generate primes < n using Sieve of Eratosthenes
def primes_up_to(n):
    sieve = np.ones(n//2, dtype=bool)
    for i in range(3, int(n**0.5) + 1, 2):
        if sieve[i//2]:
            sieve[i*i//2::i] = False
    return np.r_[2, 2*np.nonzero(sieve)[0][1::] + 1]


if __name__ == "__main__":
    # Get upper bound from command line argument
    arg = sys.argv[1]
    upper_bound = int(arg)

    # Generate primes to randomly select from
    primes = primes_up_to(upper_bound)[2:]  # Skip 2 and 3 as they are too small
    
    # Run trials
    num_trials = 100
    encrypt_times, decrypt_times, break_times = benchmark_rsa(primes, num_trials)

    # Store results in a .csv file
    with open(f"benchmark_results_upper={upper_bound}.csv", "w") as f:
        f.write("EncryptTime,DecryptTime,BreakTime\n")
        for i in range(num_trials):
            f.write(f"{encrypt_times[i]},{decrypt_times[i]},{break_times[i]}\n")

