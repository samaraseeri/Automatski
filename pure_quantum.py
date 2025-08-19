#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nonlinear BVP -> QUBO (Automatski) via Chebyshev collocation + Picard iterations.

PDE on [0,2]:
    d/dx [ (u+1) * u' ] = f(x),   u(0)=alpha,  u(2)=beta
Exact test (for diagnostics):
    u_exact(x) = exp(x) + 1
    f(x) = exp(x) * (exp(x) + x + 1)  (only used for plotting / reference)

Inner (annealer) problem is QUADRATIC because we freeze a(x,u) = a_prev = (u_prev+1)
=> r = D ( diag(a_prev) (D u) ) - f  is linear in u
=> L = || r ||_W^2 is quadratic in binary-encoded unknowns.

BCs are enforced exactly by parameterizing:
    u = u_base + S * g,  with  u_base linear matching BCs,  S = diag(s*(1-s))
so endpoints vanish automatically.

Small sizes by default: N=17, B=3 -> 51 binaries (good for demos).
"""

import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime

# -------------------------
# Automatski client classes (from your snippet)
# -------------------------
class AutomatskiInitiumTabuSolver:
    def __init__(self, host, port, max_iter=1000, tabu_tenure=10, timeout=3600):
        self.host = host; self.port = port
        self.max_iter = max_iter; self.tabu_tenure = tabu_tenure; self.timeout = timeout
    def solve(self, quboDict, silent=False):
        self.silent = silent; self.keysToIndex = {}; self.indexToKeys = {}; self.count = 0
        qubo = []
        for key in quboDict:
            m = self.index(key[0]); n = self.index(key[1])
            i = min(m, n); j = max(m, n)
            v = float(quboDict[key]); qubo.append([i, j, v])
        if not self.silent:
            print("Executing Annealer (Tabu) with ...")
            print(f"{len(self.indexToKeys.keys())} Qubits and {len(qubo)} clauses")
        tstart = datetime.datetime.now()
        r = requests.post(f'http://{self.host}:{self.port}/api/tabu',
                          json={'max_iter': self.max_iter, 'tabu_tenure': self.tabu_tenure,
                                'timeout': self.timeout, 'qubo': qubo}, timeout=None)
        tend = datetime.datetime.now()
        if not self.silent: print(f"Time Taken {(tend - tstart)}")
        struct = r.json()
        cannotProceed = None
        try:
            if struct["error"]:
                print(struct["error"]); cannotProceed = True; raise Exception(struct["error"])
        except Exception as e:
            if cannotProceed: raise e
            pass
        bits = struct['bits']; value = struct['value']; answer = {}
        for bit in bits:
            index = int(bit); answer[self.indexToKeys[index]] = bits[bit]
        self.keysToIndex = {}; self.indexToKeys = {}; self.count = 0
        return answer, value
    def index(self, key):
        if key in self.keysToIndex: return self.keysToIndex[key]
        self.keysToIndex[key] = self.count; self.indexToKeys[self.count] = key
        self.count += 1; return self.count - 1

class AutomatskiInitiumSASolver:
    def __init__(self, host, port, max_iter=1000, temp=10.0, cooling_rate=0.01, num_reads=10, timeout=3600):
        self.host = host; self.port = port
        self.max_iter = max_iter; self.temp = temp; self.cooling_rate = cooling_rate
        self.num_reads = num_reads; self.timeout = timeout
    def solve(self, quboDict, silent=False):
        self.silent = silent; self.keysToIndex = {}; self.indexToKeys = {}; self.count = 0
        qubo = []
        for key in quboDict:
            m = self.index(key[0]); n = self.index(key[1])
            i = min(m, n); j = max(m, n)
            v = float(quboDict[key]); qubo.append([i, j, v])
        if not self.silent:
            print("Executing Annealer (SA) with ...")
            print(f"{len(self.indexToKeys.keys())} Qubits and {len(qubo)} clauses")
        tstart = datetime.datetime.now()
        r = requests.post(f'http://{self.host}:{self.port}/api/sa',
                          json={'max_iter': self.max_iter, 'temp': self.temp,
                                'num_reads': self.num_reads, 'cooling_rate': self.cooling_rate,
                                'timeout': self.timeout, 'qubo': qubo})
        tend = datetime.datetime.now()
        if not self.silent: print(f"Time Taken {(tend - tstart)}")
        struct = r.json()
        cannotProceed = None
        try:
            if struct["error"]:
                print(struct["error"]); cannotProceed = True; raise Exception(struct["error"])
        except Exception as e:
            if cannotProceed: raise e
            pass
        bits = struct['bits']; value = struct['value']; answer = {}
        for bit in bits:
            index = int(bit); answer[self.indexToKeys[index]] = bits[bit]
        self.keysToIndex = {}; self.indexToKeys = {}; self.count = 0
        return answer, value
    def index(self, key):
        if key in self.keysToIndex: return self.keysToIndex[key]
        self.keysToIndex[key] = self.count; self.indexToKeys[self.count] = key
        self.count += 1; return self.count - 1

# -------------------------
# Chebyshev collocation utilities
# -------------------------
def cheb_nodes(N, a=0.0, b=2.0):
    """CGL nodes mapped to [a,b], returned from right to left (j=N..0)."""
    k = np.arange(N - 1, -1, -1)
    x_ref = np.cos(np.pi * k / (N - 1))
    return 0.5*(a+b) + 0.5*(b-a)*x_ref

def cheb_D(N, a=0.0, b=2.0):
    if N == 1: return np.array([[0.0]])
    x = cheb_nodes(N, a, b)
    c = np.hstack([2.0, np.ones(N - 2), 2.0]) * ((-1.0) ** np.arange(N))
    X = np.tile(x, (N, 1))
    dX = X - X.T + np.eye(N)
    D = (np.outer(c, 1.0 / c)) / dX
    D = D - np.diag(np.sum(D, axis=1))
    # scale for [a,b]
    D *= 2.0 / (b - a)
    return D

def clenshaw_curtis_weights(N, a=0.0, b=2.0):
    """Clenshaw–Curtis weights for CGL nodes. Scaled to [a,b]."""
    # on [-1,1], weights sum to 2. On [a,b], multiply by (b-a)/2 = 1 (since b-a=2).
    n = N - 1
    w = np.zeros(N)
    if n == 0:
        w[0] = 2.0
        return w * (b - a) / 2.0
    c = np.zeros(n + 1)
    c[0] = 2.0
    c[1::2] = 0.0
    for k in range(2, n + 1, 2):
        c[k] = 2.0 / (1 - k * k)
    # weights via cosine transform
    theta = np.arange(0, N) * np.pi / n
    w = np.zeros(N)
    for j in range(N):
        cos_kj = np.cos(np.arange(0, n + 1, 1) * theta[j])
        w[j] = (2.0 / n) * np.sum(c * cos_kj)
    return w * (b - a) / 2.0  # here (b-a)/2 = 1 for [0,2]

# -------------------------
# QUBO builder (Picard linearization)
# -------------------------
def build_picard_qubo(x, D, w_cc, a_prev, f_vec,
                      B=3, g_min=-1.0, g_max=1.0):
    """
    Build QUBO (Q, l, const) for fixed a_prev = u_prev + 1 (vector of length N).
    Encoding: g_j = g_min + step * sum_{m=0}^{B-1} 2^m b_{jm}
    u = u_base + S g  (BC exact)
    Residual: r = D (diag(a_prev) (D u)) - f
    Objective: L = r^T W r

    Returns:
      qubo_dict: {(key_i, key_j): coeff} with i<=j
      decode_info: dict with shapes and step to reconstruct g,u
    """
    N = len(x)
    # exact-BC parameterization
    s = (x - x[-1]) / (x[0] - x[-1])  # maps from decreasing x to s in [0,1] (works for CGL order)
    # Better: just map x from [0,2] to s in [0,1] explicitly:
    s = (x - 0.0) / 2.0
    u_base = (1.0 - s) * alpha + s * beta
    S = np.diag(s * (1.0 - s))  # zero at endpoints -> BC exact

    # binary encoding weights
    # g_j in [g_min, g_max], step to spread across 2^B - 1 levels
    step = (g_max - g_min) / (2**B - 1)
    Nbits = N * B

    # Build per-bit "shape vectors" phi_k so that:
    #   u = c0 + sum_k phi_k * b_k
    # where c0 = u_base + S * (g_min * 1)
    one = np.ones(N)
    c0 = u_base + S @ (g_min * one)  # offset due to g_min

    phi = []   # list of vectors in R^N, one per bit
    keys = []  # labels for bits (node j, bit m)
    for j in range(N):
        for m in range(B):
            weight = step * (2**m)
            vec = S[:, j] * weight  # column j of S times weight
            phi.append(vec)
            keys.append(("b", j, m))
    phi = np.stack(phi, axis=1)  # N x Nbits

    # v = D u = D c0 + sum_k (D phi_k) b_k
    Dc0 = D @ c0
    Dphi = D @ phi  # N x Nbits

    # r = D ( diag(a_prev) v ) - f
    A1 = D @ (np.diag(a_prev) @ Dphi)   # N x Nbits
    c_r = D @ (np.diag(a_prev) @ Dc0) - f_vec  # N

    # Weighted least squares: L = (c_r + A1 b)^T W (c_r + A1 b)
    W = np.diag(w_cc)
    # Precompute
    Q = A1.T @ W @ A1                 # (Nbits x Nbits)
    l = 2.0 * (A1.T @ W @ c_r)        # (Nbits,)
    const = float(c_r.T @ W @ c_r)    # scalar

    # Optional: small Tikhonov smoothing on u to stabilize:
    # lambda_s * || D^2 u ||^2
    lambda_s = 0.0  # set >0 if needed
    if lambda_s > 0.0:
        D2 = D @ D
        c_D2 = D2 @ c0
        A_D2 = D2 @ phi
        Q += lambda_s * (A_D2.T @ A_D2)
        l += 2.0 * lambda_s * (A_D2.T @ c_D2)
        const += lambda_s * float(c_D2.T @ c_D2)

    # Assemble QUBO dict: sum_{i<=j} Q_ij b_i b_j + sum_i l_i b_i + const
    qubo = {}
    # quadratic off-diagonals and diag from Q
    for i in range(Nbits):
        # diagonal includes Q_ii + linear l_i
        coeff_ii = Q[i, i] + l[i]
        if abs(coeff_ii) > 0.0:
            qubo[(keys[i], keys[i])] = qubo.get((keys[i], keys[i]), 0.0) + float(coeff_ii)
        for j in range(i + 1, Nbits):
            qij = Q[i, j]
            if abs(qij) > 0.0:
                qubo[(keys[i], keys[j])] = qubo.get((keys[i], keys[j]), 0.0) + float(qij)

    decode_info = {
        "keys": keys, "N": N, "B": B, "g_min": g_min, "step": step,
        "S_diag": np.diag(S), "u_base": u_base, "const": const
    }
    return qubo, decode_info

def bits_to_u(answer_bits, decode_info):
    """Decode Automatski bit dict -> (g,u) on nodes."""
    keys = decode_info["keys"]; N = decode_info["N"]; B = decode_info["B"]
    g_min = decode_info["g_min"]; step = decode_info["step"]
    S_diag = decode_info["S_diag"]; u_base = decode_info["u_base"]

    # collect bits per (j,m)
    g = np.zeros(N)
    for j in range(N):
        acc = 0.0
        for m in range(B):
            key = ("b", j, m)
            bit = int(answer_bits.get(key, 0))
            acc += (2**m) * bit
        g[j] = g_min + step * acc
    u = u_base + S_diag * g
    return g, u

# -------------------------
# Test RHS and exact solution (for plotting/diagnostics)
# -------------------------
def f_rhs(x):
    return np.exp(x) * (np.exp(x) + x + 1.0)

def u_exact(x):
    return np.exp(x) + 1.0

# -------------------------
# Main driver
# -------------------------
if __name__ == "__main__":
    # Domain and BCs
    a_dom, b_dom = 0.0, 2.0
    alpha, beta = u_exact(a_dom), u_exact(b_dom)

    # Collocation
    N = 17                   # nodes (keep modest)
    x = cheb_nodes(N, a_dom, b_dom)
    D = cheb_D(N, a_dom, b_dom)
    w_cc = clenshaw_curtis_weights(N, a_dom, b_dom)
    f_vec = f_rhs(x)

    # Binary encoding (per node)
    B = 3        # bits per node (try 3–4)
    g_min, g_max = -1.0, 1.0

    # Automatski endpoint
    HOST = "localhost"   # <-- set your host
    PORT = 5000          # <-- set your port
    USE_TABU = True      # or False to use SA

    if USE_TABU:
        solver = AutomatskiInitiumTabuSolver(HOST, PORT, max_iter=2000, tabu_tenure=10, timeout=600)
    else:
        solver = AutomatskiInitiumSASolver(HOST, PORT, max_iter=2000, temp=10.0, cooling_rate=0.01, num_reads=20, timeout=600)

    # Picard outer loop
    max_picard = 5
    u_prev = u_exact(x) * 0 + ( (1.0 - (x - a_dom)/(b_dom - a_dom)) * alpha + ((x - a_dom)/(b_dom - a_dom)) * beta )  # linear init

    history = []
    for it in range(1, max_picard + 1):
        a_prev = u_prev + 1.0
        qubo, info = build_picard_qubo(x, D, w_cc, a_prev, f_vec, B=B, g_min=g_min, g_max=g_max)

        print(f"[Picard {it}] QUBO vars = {len(info['keys'])}, terms = {len(qubo)}")
        answer_bits, energy = solver.solve(qubo, silent=False)
        g_hat, u_hat = bits_to_u(answer_bits, info)

        # residual norm
        r = D @ (np.diag(a_prev) @ (D @ u_hat)) - f_vec
        l2 = float(np.sqrt(np.sum(w_cc * r * r)))
        history.append((it, l2, energy))
        print(f"[Picard {it}] Weighted residual L2 ≈ {l2:.3e}, Annealer energy = {energy:.6e}")

        # update
        u_prev = u_hat.copy()

    # Plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(x, u_exact(x), label="Exact $u(x)=e^x+1$", linewidth=2)
    plt.plot(x, u_prev, "o-", label="Annealer (Picard last iterate)")
    plt.gca().invert_xaxis()  # CGL nodes are returned right->left; flipping is visually natural
    plt.grid(True, linestyle=":")
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.title(f"Nonlinear BVP via QUBO (N={N}, B={B}, Picard iters={max_picard})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qubo_bvp_automatski.png", dpi=200)
    plt.savefig("qubo_bvp_automatski.pdf")
    print("[Plot] Saved to qubo_bvp_automatski.png and qubo_bvp_automatski.pdf")

    print("\n[Picard history]")
    for it, l2, en in history:
        print(f"  iter {it:2d}:  ||r||_W ≈ {l2:.4e} ,  energy = {en:.6e}")
