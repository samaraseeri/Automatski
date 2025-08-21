#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Picard + QUBO Annealer for a nonlinear BVP:
    d/dx ( (u+1) * du/dx ) = f(x),  x in [A,B],  u(A)=alpha, u(B)=beta.

- Discretize with Chebyshev–Gauss–Lobatto (CGL) collocation.
- Enforce BCs exactly via u(x) = u_base(x) + S(x) * g(x), where
  u_base is the linear function satisfying BCs and S(x)=(x-A)(B-x) vanishes at endpoints.
- Picard linearization: freeze a(u) at a(u_prev) so residual is linear in u.
- Minimize || D diag(a(u_prev)) D u - f ||^2 over g (quadratic in g).
- Quantize g with B-bit uniform encoding and build a QUBO.
- Solve QUBO on Automatski (tabu or simulated annealing endpoint).
- Iterate Picard until residual decreases / convergence.
- Save plot comparing Exact vs Hybrid Annealer solution.

Requirements:
- numpy, matplotlib, requests
"""

import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime

# ---------------------------
# Automatski clients
# ---------------------------

class AutomatskiInitiumTabuSolver:
    def __init__(self, host, port, max_iter=1000, tabu_tenure=10, timeout=3600):
        self.host = host
        self.port = port
        self.max_iter= max_iter
        self.tabu_tenure= tabu_tenure
        self.timeout = timeout

    def solve(self, quboDict, silent=False):
        self.silent = silent
        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0

        qubo = []
        for key in quboDict:
            m = self.index(key[0])
            n = self.index(key[1])
            i = min(m,n)
            j = max(m,n)
            v = float(quboDict[key])
            qubo.append([i,j,v])

        if not self.silent:
            print("Executing Annealer (Tabu) with ...")
            print(f"{len(self.indexToKeys.keys())} Qubits and {len(qubo)} clauses")

        tstart = datetime.datetime.now()
        r = requests.post(
            f'http://{self.host}:{self.port}/api/tabu',
            json={'max_iter': self.max_iter, 'tabu_tenure': self.tabu_tenure,
                  'timeout': self.timeout, 'qubo': qubo},
            timeout=None
        )
        tend = datetime.datetime.now()
        if not self.silent:
            print(f"Time Taken {(tend - tstart)}")

        struct = r.json()
        if isinstance(struct, dict) and "error" in struct and struct["error"]:
            raise RuntimeError(struct["error"])

        bits = struct['bits']
        value = struct['value']
        answer = {}
        for bit in bits:
            index = int(bit)
            answer[self.indexToKeys[index]] = bits[bit]

        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0
        return answer, value

    def index(self, key):
        if key in self.keysToIndex:
            return self.keysToIndex[key]
        self.keysToIndex[key] = self.count
        self.indexToKeys[self.count] = key
        self.count += 1
        return self.count - 1


class AutomatskiInitiumSASolver:
    def __init__(self, host, port, max_iter=1000, temp=10.0, cooling_rate=0.01, num_reads=10, timeout=3600):
        self.host = host
        self.port = port
        self.max_iter= max_iter
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.num_reads = num_reads
        self.timeout = timeout

    def solve(self, quboDict, silent=False):
        self.silent = silent
        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0

        qubo = []
        for key in quboDict:
            m = self.index(key[0])
            n = self.index(key[1])
            i = min(m,n)
            j = max(m,n)
            v = float(quboDict[key])
            qubo.append([i,j,v])

        if not self.silent:
            print("Executing Annealer (SA) with ...")
            print(f"{len(self.indexToKeys.keys())} Qubits and {len(qubo)} clauses")

        tstart = datetime.datetime.now()
        r = requests.post(
            f'http://{self.host}:{self.port}/api/sa',
            json={'max_iter': self.max_iter, 'temp': self.temp, 'num_reads': self.num_reads,
                  'cooling_rate': self.cooling_rate, 'timeout': self.timeout, 'qubo': qubo}
        )
        tend = datetime.datetime.now()
        if not self.silent:
            print(f"Time Taken {(tend - tstart)}")

        struct = r.json()
        if isinstance(struct, dict) and "error" in struct and struct["error"]:
            raise RuntimeError(struct["error"])

        bits = struct['bits']
        value = struct['value']
        answer = {}
        for bit in bits:
            index = int(bit)
            answer[self.indexToKeys[index]] = bits[bit]

        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0
        return answer, value

    def index(self, key):
        if key in self.keysToIndex:
            return self.keysToIndex[key]
        self.keysToIndex[key] = self.count
        self.indexToKeys[self.count] = key
        self.count += 1
        return self.count - 1

# ---------------------------
# Chebyshev utilities
# ---------------------------

def chebyshev_nodes(N, a, b):
    """Chebyshev-Gauss-Lobatto nodes mapped to [a,b]."""
    if N == 1:
        return np.array([(a+b)/2.0])
    k = np.arange(N-1, -1, -1)
    x_ref = np.cos(np.pi * k / (N-1))
    return 0.5*(a+b) + 0.5*(b-a)*x_ref

def chebyshev_differentiation_matrix(N, a, b):
    """CGL differentiation matrix on [a,b]."""
    if N == 1:
        return np.array([[0.0]])
    x = chebyshev_nodes(N, a, b)
    c = np.hstack([2.0, np.ones(N-2), 2.0]) * ((-1.0) ** np.arange(N))
    X = np.tile(x, (N,1))
    dX = X - X.T + np.eye(N)
    D = (np.outer(c, 1.0/c)) / dX
    D = D - np.diag(np.sum(D, axis=1))
    D *= 2.0 / (b - a)
    return D

def shape_function(x, A, B):
    """S(x) that vanishes at A and B; smooth in (A,B)."""
    return (x - A)*(B - x)

def u_base_linear(x, A, B, alpha, beta):
    """Linear function matching the Dirichlet BCs exactly."""
    t = (x - A) / (B - A)
    return alpha*(1.0 - t) + beta*t

# ---------------------------
# Problem setup
# ---------------------------

A, B = 0.0, 2.0
alpha = 2.0
beta  = float(np.exp(2.0) + 1.0)  # matches exact u(x)=e^x+1 at x=2

# Nonlinearity and forcing consistent with exact u = e^x + 1
def a_u(u):      # a(u) = u + 1
    return u + 1.0

def u_exact(x):
    return np.exp(x) + 1.0

def f_rhs(x):    # f(x) for exact solution u = e^x + 1
    return 2.0 * np.exp(2.0 * x) + 2.0 * np.exp(x)

# ---------------------------
# QUBO builder (Picard step)
# ---------------------------

def build_picard_qubo(u_prev, x, D, A, B, alpha, beta,
                      g_min=-2.0, g_max=2.0, Bbits=3,
                      lambda_reg=0.0, weight_diag=None,
                      prune_tol=1e-12):
    """
    Build QUBO for minimizing || D diag(a(u_prev)) D u - f ||^2
    with u = u_base + S*g, and g quantized in [g_min, g_max] with Bbits per node.

    Returns:
      qubo_dict: {(var_i, var_j): coeff}
      var_keys:  ordered list of variable keys for decoding
      P, q0:     mapping s.t. g ≈ P z + q0 (z ∈ {0,1}^{N*B})
      M, d, W:   matrices for diagnostics (residual structure)
    """
    N = len(x)
    S = shape_function(x, A, B)            # (N,)
    Ubase = u_base_linear(x, A, B, alpha, beta)
    a_prev = a_u(u_prev)                   # (N,)
    Aop = D @ (np.diag(a_prev) @ D)        # (N x N)

    # Residual r(u) = Aop u - f
    # Substitute u = Ubase + S*g => r = M g + d
    M = Aop @ np.diag(S)                   # (N x N)
    d = Aop @ Ubase - f_rhs(x)             # (N,)

    # Weights (optional): identity by default
    if weight_diag is None:
        W = np.eye(N)
    else:
        W = np.diag(weight_diag)

    # Quadratic form in g: J(g) = (Mg + d)^T W (Mg + d) = g^T Q g + 2 c^T g + const
    Q = M.T @ W @ M                        # (N x N)
    c = M.T @ W @ d                        # (N,)

    # Small Tikhonov regularization on g (optional)
    if lambda_reg > 0.0:
        Q = Q + lambda_reg * np.eye(N)
        c = c + 0.5 * lambda_reg * np.zeros_like(c)

    # ----- Binary encoding: g ≈ q0 + P z -----
    # Uniform unsigned B-bit encoding per node in [g_min, g_max].
    # Δ = (g_max - g_min)/(2^B - 1);  g_j = g_min + Δ * sum_{b=0}^{B-1} 2^b z_{j,b}
    levels = 2**Bbits - 1
    Δ = (g_max - g_min) / levels
    bit_weights = np.array([2**b for b in range(Bbits)], dtype=float)  # (Bbits,)

    # Build P (N x NB) and q0 (N,)
    NB = N * Bbits
    P = np.zeros((N, NB), dtype=float)
    var_keys = []
    for j in range(N):
        for b in range(Bbits):
            col = j*Bbits + b
            P[j, col] = Δ * bit_weights[b]
            var_keys.append(('g', j, b))
    q0 = g_min * np.ones(N)

    # Compose QUBO in z: J(z) = z^T (P^T Q P) z + 2 (P^T (Q q0 + c))^T z + const
    Qz = P.T @ Q @ P
    lin = 2.0 * (P.T @ (Q @ q0 + c))  # linear vector; fold into diag

    # Fold linear terms into diagonal (since z_i^2=z_i)
    for i in range(NB):
        Qz[i, i] += lin[i]

    # Prune tiny coefficients for compactness
    qubo_dict = {}
    for i in range(NB):
        for j in range(i, NB):
            v = Qz[i, j]
            if abs(v) > prune_tol:
                key = (var_keys[i], var_keys[j])
                qubo_dict[key] = float(v)

    return qubo_dict, var_keys, P, q0, M, d, W

# ---------------------------
# Decoding & residuals
# ---------------------------

def decode_bits_to_g(answer, var_keys, N, Bbits, P, q0):
    """Build z (0/1), g ≈ P z + q0 from annealer's answer dict."""
    NB = N * Bbits
    z = np.zeros(NB, dtype=float)
    idx = {var_keys[k]: k for k in range(NB)}
    for k, v in answer.items():
        if k in idx:
            z[idx[k]] = 1.0 if int(v) == 1 else 0.0
    g = q0 + P @ z
    return g, z

def residual_L2(u, x, D):
    r = D @ (a_u(u) * (D @ u)) - f_rhs(x)
    return float(np.linalg.norm(r, 2)), r

# ---------------------------
# Main: Picard + Annealer loop
# ---------------------------

def run_hybrid_picard_annealer(
        host="localhost", port=8080, use_tabu=True,
        N=17, Bbits=3, g_min=-2.0, g_max=2.0,
        max_picard=8, gamma=0.6,  # damping for updates
        lambda_reg=0.0,
        prune_tol=1e-12,
        plot_file_png="qubo_bvp_hybrid_automatski.png",
        plot_file_pdf="qubo_bvp_hybrid_automatski.pdf"):

    print("[Config] Host/Port:", host, port)
    print(f"[Config] N={N}, Bbits={Bbits}, g_range=[{g_min},{g_max}], max_picard={max_picard}, gamma={gamma}")

    # Discretization
    x = chebyshev_nodes(N, A, B)
    D = chebyshev_differentiation_matrix(N, A, B)

    # Initial guess: exact BC line
    u_prev = u_base_linear(x, A, B, alpha, beta)

    # Choose annealer client
    if use_tabu:
        annealer = AutomatskiInitiumTabuSolver(host, port, max_iter=2000, tabu_tenure=25, timeout=3600)
    else:
        annealer = AutomatskiInitiumSASolver(host, port, max_iter=2000, temp=10.0, cooling_rate=0.01, num_reads=10, timeout=3600)

    history = []
    for k in range(1, max_picard+1):
        # Build QUBO at current u_prev
        qubo_dict, var_keys, P, q0, M, d, W = build_picard_qubo(
            u_prev, x, D, A, B, alpha, beta,
            g_min=g_min, g_max=g_max, Bbits=Bbits,
            lambda_reg=lambda_reg, weight_diag=None, prune_tol=prune_tol
        )

        Nvars = len(var_keys)
        Nterms = len(qubo_dict)
        print(f"[Picard {k}] QUBO vars = {Nvars}, terms = {Nterms}")

        # Solve on Automatski
        answer, energy = annealer.solve(qubo_dict, silent=False)

        # Decode to g and update u
        g, z = decode_bits_to_g(answer, var_keys, N, Bbits, P, q0)
        S = shape_function(x, A, B)
        u_hat = u_base_linear(x, A, B, alpha, beta) + S * g

        # Damped update
        u_next = (1.0 - gamma) * u_prev + gamma * u_hat

        # Diagnostics
        nrm, rvec = residual_L2(u_next, x, D)
        history.append((k, nrm, energy))
        print(f"[Picard {k}] ||r||_2 ≈ {nrm:.3e}, annealer energy = {energy:.6e}")

        # Convergence check (relative decrease or absolute)
        if k > 1:
            prev = history[-2][1]
            if nrm < 1e-6 or (prev > 0 and (prev - nrm)/prev < 5e-3):
                print(f"[Stop ] Converged (or stagnated) at iter {k}.")
                u_prev = u_next
                break

        u_prev = u_next

    print("\n[Picard history]")
    for it, nrm, en in history:
        print(f"  iter {it:2d}:  ||r||_2 ≈ {nrm:.6e} ,  energy = {en:.6e}")

    # ---- Plot (Exact vs Hybrid only) ----
    plt.figure(figsize=(9.0, 5.6))
    plt.plot(x, u_exact(x), label="Exact  $u(x)=e^x+1$", lw=2.0)
    plt.plot(x, u_prev, "-o", label="Hybrid (Picard + QUBO annealer)", ms=4)
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.title("Nonlinear BVP via Hybrid Picard + QUBO Annealer")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file_png, dpi=200)
    plt.savefig(plot_file_pdf)
    print(f"[Plot] Saved to {plot_file_png} and {plot_file_pdf}")

    # Optional: save arrays
    try:
        np.save("x_nodes.npy", x)
        np.save("u_hybrid_final.npy", u_prev)
        np.save("u_exact_on_nodes.npy", u_exact(x))
    except Exception:
        pass

    return x, u_prev, history


if __name__ == "__main__":
    # ---- User knobs ----
    HOST = "localhost"   # <-- set your Automatski host
    PORT = 8080          # <-- set your Automatski port
    USE_TABU = True      # True=tabu, False=simulated annealing

    # Discretization & encoding
    N_nodes   = 17       # try 17–21
    B_bits    = 3        # try 3 or 4 before increasing N
    G_MIN, G_MAX = -2.0, 2.0

    # Picard loop
    MAX_PICARD = 6
    GAMMA = 0.6          # damping (0.3–0.8)
    LAMBDA_REG = 0.0     # small Tikhonov on g, e.g., 1e-6 if needed

    run_hybrid_picard_annealer(
        host=HOST, port=PORT, use_tabu=USE_TABU,
        N=N_nodes, Bbits=B_bits, g_min=G_MIN, g_max=G_MAX,
        max_picard=MAX_PICARD, gamma=GAMMA, lambda_reg=LAMBDA_REG
    )
