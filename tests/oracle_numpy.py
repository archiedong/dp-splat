"""Brute-force NumPy oracle for DP-Splat CAVI updates.

Independent ground-truth implementation of the update equations in
DP_SPLAT_RESEARCH_BRIEF.md (sections 3.2, 3.4, 3.5, 3.7, Appendix A/B).
Written ONLY from the brief -- deliberately NOT from src/dp_splat -- so that
bugs in the main implementation cannot correlate with bugs here.

Style is intentionally naive: explicit Python loops over n and k, no
vectorization tricks. Problems are tiny (N <= 200, T <= 10, D <= 3), so
clarity beats speed.

Dependencies: numpy, scipy.special (digamma only; gammaln imported for
completeness per the allowed dependency list).
"""

import numpy as np
from scipy.special import digamma, gammaln  # noqa: F401  (gammaln allowed)


def soft_stats(x, r):
    """Soft counts and weighted moments, brief section 3.4 preamble.

        N_k    = sum_n r_nk
        xbar_k = (1/N_k) sum_n r_nk x_n
        S_k    = sum_n r_nk (x_n - xbar_k)(x_n - xbar_k)^T   (centered scatter)

    x: (N, D), r: (N, K).
    Returns (Nk (K,), xbar (K, D), S (K, D, D)).
    For an empty component (N_k == 0) the xbar row and S slab are zeros.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    N, D = x.shape
    K = r.shape[1]

    Nk = np.zeros(K)
    xbar = np.zeros((K, D))
    S = np.zeros((K, D, D))

    for k in range(K):
        for n in range(N):
            Nk[k] += r[n, k]
        if Nk[k] == 0.0:
            continue  # leave xbar[k], S[k] as zeros
        for n in range(N):
            xbar[k] += r[n, k] * x[n]
        xbar[k] /= Nk[k]
        for n in range(N):
            d = x[n] - xbar[k]
            S[k] += r[n, k] * np.outer(d, d)
    return Nk, xbar, S


def niw_update(m0, kappa0, Psi0, nu0, Nk, xbar, S):
    """NIW posterior updates, brief eq. (1), per modality, looped over k.

        kappa_k = kappa0 + N_k
        m_k     = (kappa0 m0 + N_k xbar_k) / kappa_k
        nu_k    = nu0 + N_k
        Psi_k   = Psi0 + S_k
                  + (kappa0 N_k)/(kappa0 + N_k) (xbar_k - m0)(xbar_k - m0)^T

    m0: (D,), kappa0: scalar, Psi0: (D, D), nu0: scalar,
    Nk: (K,), xbar: (K, D), S: (K, D, D).
    Returns dict(m=(K, D), kappa=(K,), Psi=(K, D, D), nu=(K,)).
    """
    m0 = np.asarray(m0, dtype=float)
    Psi0 = np.asarray(Psi0, dtype=float)
    Nk = np.asarray(Nk, dtype=float)
    xbar = np.asarray(xbar, dtype=float)
    S = np.asarray(S, dtype=float)
    K, D = xbar.shape

    m = np.zeros((K, D))
    kappa = np.zeros(K)
    Psi = np.zeros((K, D, D))
    nu = np.zeros(K)

    for k in range(K):
        kappa[k] = kappa0 + Nk[k]
        m[k] = (kappa0 * m0 + Nk[k] * xbar[k]) / kappa[k]
        nu[k] = nu0 + Nk[k]
        d = xbar[k] - m0
        Psi[k] = Psi0 + S[k] + (kappa0 * Nk[k]) / (kappa0 + Nk[k]) * np.outer(d, d)
    return dict(m=m, kappa=kappa, Psi=Psi, nu=nu)


def expected_logdet_precision(Psi, nu):
    """E[log|Lambda_k|], brief eq. (4) first identity, with W_k := Psi_k^{-1}.

        E[log|Lambda_k|] = sum_{i=1}^{D} psi((nu_k + 1 - i)/2)
                           + D log 2 + log|W_k|

    Psi: (K, D, D), nu: (K,). Returns (K,).
    """
    Psi = np.asarray(Psi, dtype=float)
    nu = np.asarray(nu, dtype=float)
    K, D, _ = Psi.shape

    out = np.zeros(K)
    for k in range(K):
        W = np.linalg.inv(Psi[k])
        sign, logdetW = np.linalg.slogdet(W)
        term = 0.0
        for i in range(1, D + 1):
            term += digamma((nu[k] + 1 - i) / 2.0)
        out[k] = term + D * np.log(2.0) + logdetW
    return out


def expected_mahalanobis(m, kappa, Psi, nu, x):
    """E[(x - mu_k)^T Lambda_k (x - mu_k)], brief eq. (4) second identity.

        E[...] = D / kappa_k + nu_k (x - m_k)^T W_k (x - m_k),  W_k = Psi_k^{-1}

    m: (K, D), kappa: (K,), Psi: (K, D, D), nu: (K,), x: (N, D).
    Explicit double loop over n and k. Returns (N, K).
    """
    m = np.asarray(m, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    Psi = np.asarray(Psi, dtype=float)
    nu = np.asarray(nu, dtype=float)
    x = np.asarray(x, dtype=float)
    N, D = x.shape
    K = m.shape[0]

    out = np.zeros((N, K))
    for k in range(K):
        W = np.linalg.inv(Psi[k])
        for n in range(N):
            d = x[n] - m[k]
            out[n, k] = D / kappa[k] + nu[k] * float(d @ W @ d)
    return out


def expected_gauss_loglik(m, kappa, Psi, nu, x):
    """Per-modality bracket of brief eq. (5):

        0.5 E[log|Lambda_k|] - (D/2) log(2 pi)
        - 0.5 E[(x_n - mu_k)^T Lambda_k (x_n - mu_k)]

    m: (K, D), kappa: (K,), Psi: (K, D, D), nu: (K,), x: (N, D).
    Returns (N, K).
    """
    x = np.asarray(x, dtype=float)
    N, D = x.shape
    K = np.asarray(m).shape[0]

    elogdet = expected_logdet_precision(Psi, nu)  # (K,)
    emaha = expected_mahalanobis(m, kappa, Psi, nu, x)  # (N, K)

    out = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            out[n, k] = (0.5 * elogdet[k]
                         - 0.5 * D * np.log(2.0 * np.pi)
                         - 0.5 * emaha[n, k])
    return out


def dp_update(Nk, e_alpha):
    """Stick-breaking Beta updates, brief eq. (2) (Variant A).

        gamma_{k,1} = 1 + N_k
        gamma_{k,2} = E_q[alpha] + sum_{j=k+1}^{T} N_j        (k = 1..T-1)

    Nk: (T,), e_alpha: scalar E_q[alpha].
    Returns (gamma1 (T-1,), gamma2 (T-1,)). Tail sums done explicitly.
    """
    Nk = np.asarray(Nk, dtype=float)
    T = Nk.shape[0]

    gamma1 = np.zeros(T - 1)
    gamma2 = np.zeros(T - 1)
    for k in range(T - 1):  # 0-based k corresponds to brief's k = 1..T-1
        gamma1[k] = 1.0 + Nk[k]
        tail = 0.0
        for j in range(k + 1, T):  # brief: j = k+1 .. T
            tail += Nk[j]
        gamma2[k] = e_alpha + tail
    return gamma1, gamma2


def dp_elogpi(gamma1, gamma2):
    """Expected log stick-breaking weights, brief eq. (3).

        E[log v_k]     = psi(gamma_{k,1}) - psi(gamma_{k,1} + gamma_{k,2})
        E[log(1-v_k)]  = psi(gamma_{k,2}) - psi(gamma_{k,1} + gamma_{k,2})
        E[log pi_k]    = E[log v_k] + sum_{j<k} E[log(1-v_j)]

    with v_T := 1 (brief section 3.2), hence E[log v_T] = 0.
    gamma1, gamma2: (T-1,). Returns (T,).
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    Tm1 = gamma1.shape[0]
    T = Tm1 + 1

    elogv, elog1mv = beta_expectations(gamma1, gamma2)

    elogpi = np.zeros(T)
    for k in range(T):
        if k < Tm1:
            elogpi[k] = elogv[k]
        else:
            elogpi[k] = 0.0  # E[log v_T] = 0 since v_T := 1
        for j in range(k):  # j < k
            elogpi[k] += elog1mv[j]
    return elogpi


def dp_expected_pi(gamma1, gamma2):
    """Posterior expected stick-breaking weights, brief section 3.7 bullet 1.

        E[pi_k] = gamma_{k,1}/(gamma_{k,1}+gamma_{k,2})
                  * prod_{j<k} gamma_{j,2}/(gamma_{j,1}+gamma_{j,2})

    with v_T := 1, so E[pi_T] = prod_{j<T} gamma_{j,2}/(gamma_{j,1}+gamma_{j,2}).
    gamma1, gamma2: (T-1,). Returns (T,); sums to 1 exactly (Appendix B).
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    Tm1 = gamma1.shape[0]
    T = Tm1 + 1

    epi = np.zeros(T)
    for k in range(T):
        if k < Tm1:
            ev = gamma1[k] / (gamma1[k] + gamma2[k])  # E[v_k]
        else:
            ev = 1.0  # v_T := 1
        prod = 1.0
        for j in range(k):  # j < k
            prod *= gamma2[j] / (gamma1[j] + gamma2[j])  # E[1 - v_j]
        epi[k] = ev * prod
    return epi


def dir_update(Nk, e0):
    """Sparse/finite Dirichlet posterior (Variant B, brief section 3.2/3.4):

        alpha_post_k = e0 + N_k

    Nk: (T,), e0: scalar. Returns (T,).
    """
    Nk = np.asarray(Nk, dtype=float)
    T = Nk.shape[0]
    alpha_post = np.zeros(T)
    for k in range(T):
        alpha_post[k] = e0 + Nk[k]
    return alpha_post


def dir_elogpi(alpha_post):
    """Dirichlet expected log-weights, brief eq. (3) Variant B remark:

        E[log pi_k] = psi(alpha_k) - psi(sum_j alpha_j)

    alpha_post: (T,). Returns (T,).
    """
    alpha_post = np.asarray(alpha_post, dtype=float)
    T = alpha_post.shape[0]
    total = 0.0
    for k in range(T):
        total += alpha_post[k]
    out = np.zeros(T)
    for k in range(T):
        out[k] = digamma(alpha_post[k]) - digamma(total)
    return out


def alpha_update(elog1mv, a0, b0):
    """Gamma update for the DP concentration alpha, brief eq. (6):

        w1 = a0 + T - 1
        w2 = b0 - sum_{k=1}^{T-1} E[log(1 - v_k)]

    elog1mv: (T-1,) values of E[log(1-v_k)]. Returns (w1, w2) scalars.
    (E_q[alpha] = w1 / w2, computed by the caller.)
    """
    elog1mv = np.asarray(elog1mv, dtype=float)
    Tm1 = elog1mv.shape[0]
    w1 = a0 + Tm1  # a0 + T - 1
    s = 0.0
    for k in range(Tm1):
        s += elog1mv[k]
    w2 = b0 - s
    return w1, w2


def responsibilities(elogpi_vec, ell_list):
    """Responsibilities, brief eq. (5):

        log rho_nk = E[log pi_k] + sum_{m} [per-modality expected log-density]
        r_nk = exp(log rho_nk) / sum_j exp(log rho_nj)   (log-sum-exp)

    elogpi_vec: (K,); ell_list: list of (N, K) expected log-density arrays
    (one per modality, each already the full bracket of eq. (5)).
    Returns r (N, K) with rows summing to 1.
    """
    elogpi_vec = np.asarray(elogpi_vec, dtype=float)
    K = elogpi_vec.shape[0]
    N = np.asarray(ell_list[0]).shape[0]

    logrho = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            logrho[n, k] = elogpi_vec[k]
            for ell in ell_list:
                logrho[n, k] += ell[n, k]

    r = np.zeros((N, K))
    for n in range(N):
        mx = logrho[n, 0]
        for k in range(1, K):
            if logrho[n, k] > mx:
                mx = logrho[n, k]
        denom = 0.0
        for k in range(K):
            r[n, k] = np.exp(logrho[n, k] - mx)
            denom += r[n, k]
        for k in range(K):
            r[n, k] /= denom
    return r


def beta_expectations(gamma1, gamma2):
    """Beta log-expectations, brief Appendix B:

        E[log v]     = psi(a) - psi(a + b)
        E[log(1-v)]  = psi(b) - psi(a + b)      for v ~ Beta(a, b)

    gamma1, gamma2: (T-1,). Returns (E[log v], E[log(1-v)]), each (T-1,).
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    Tm1 = gamma1.shape[0]
    elogv = np.zeros(Tm1)
    elog1mv = np.zeros(Tm1)
    for k in range(Tm1):
        elogv[k] = digamma(gamma1[k]) - digamma(gamma1[k] + gamma2[k])
        elog1mv[k] = digamma(gamma2[k]) - digamma(gamma1[k] + gamma2[k])
    return elogv, elog1mv


if __name__ == "__main__":
    # Tiny self-check (brief Appendix B sanity identities).
    rng = np.random.default_rng(0)
    N, T, D = 50, 5, 2

    x = rng.normal(size=(N, D))
    noise = rng.normal(size=(N, T))
    # random responsibilities via row-wise softmax
    r = np.exp(noise - noise.max(axis=1, keepdims=True))
    r /= r.sum(axis=1, keepdims=True)

    Nk, xbar, S = soft_stats(x, r)
    assert abs(Nk.sum() - N) < 1e-9, "soft_stats: Nk must sum to N"

    gamma1, gamma2 = dp_update(Nk, e_alpha=1.0)
    epi = dp_expected_pi(gamma1, gamma2)
    assert abs(epi.sum() - 1.0) < 1e-12, "dp_expected_pi must sum to 1 (v_T := 1)"

    # Exercise the remaining pieces end-to-end (shape/crash check).
    post = niw_update(m0=np.zeros(D), kappa0=1e-3, Psi0=np.eye(D),
                      nu0=D + 2, Nk=Nk, xbar=xbar, S=S)
    ell = expected_gauss_loglik(post["m"], post["kappa"], post["Psi"],
                                post["nu"], x)
    elogpi = dp_elogpi(gamma1, gamma2)
    r_new = responsibilities(elogpi, [ell])
    assert np.allclose(r_new.sum(axis=1), 1.0), "responsibility rows must sum to 1"

    elogv, elog1mv = beta_expectations(gamma1, gamma2)
    w1, w2 = alpha_update(elog1mv, a0=1.0, b0=1.0)
    assert w1 == 1.0 + (T - 1) and w2 > 0.0

    alpha_post = dir_update(Nk, e0=0.01)
    _ = dir_elogpi(alpha_post)

    print("OK")
