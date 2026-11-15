"""Three-state HMM (H, A, D) with team evidence fixed and liveness emissions + transitions learned
by EM on the unlabelled match. No ground truth of any kind enters this file.

Team evidence (fixed): a log-likelihood ratio for home vs away from the ball (soft nearest player)
and from the self-supervised shape model. Liveness evidence (learned): Gaussian / Bernoulli
emissions on player tempo, ball presence, ball-gap length, ball speed, out-of-bounds; parameters
are shared by H and A ("alive") and separate for D.
"""
import numpy as np

H, A, D = 0, 1, 2
LOG_EPS = 1e-12


def logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class Liveness:
    """Per-frame liveness observations. Each channel has a mask (observed or not) and a kind."""
    def __init__(self, obs):
        self.obs = obs  # dict name -> (values, mask, kind) ; kind in {"gauss","bern"}
        self.params = {}

    def init(self, dead_prior):
        """Initialise from a soft heuristic responsibility for D (only used as a starting point)."""
        w_d = dead_prior; w_a = 1 - dead_prior
        self.m_step(np.column_stack([w_a / 2, w_a / 2, w_d]))

    def m_step(self, gamma):
        alive = gamma[:, H] + gamma[:, A]; dead = gamma[:, D]
        for name, (v, m, kind) in self.obs.items():
            p = {}
            for st, w in (("alive", alive), ("dead", dead)):
                ww = w * m
                if kind == "gauss":
                    mu = np.sum(ww * np.where(m, v, 0)) / (ww.sum() + 1e-9)
                    var = np.sum(ww * np.where(m, (v - mu) ** 2, 0)) / (ww.sum() + 1e-9)
                    p[st] = (mu, max(var, 1e-3))
                else:
                    p[st] = (np.sum(ww * np.where(m, v, 0)) + 1) / (ww.sum() + 2)
            self.params[name] = p

    def loglik(self):
        """(n, 2) log-likelihood for alive, dead."""
        n = len(next(iter(self.obs.values()))[0]); out = np.zeros((n, 2))
        for name, (v, m, kind) in self.obs.items():
            for j, st in enumerate(("alive", "dead")):
                if kind == "gauss":
                    mu, var = self.params[name][st]
                    ll = -0.5 * (np.log(2 * np.pi * var) + (v - mu) ** 2 / var)
                else:
                    q = self.params[name][st]
                    ll = np.where(v > 0.5, np.log(q), np.log(1 - q))
                out[:, j] += np.where(m, ll, 0.0)
        return out


def emissions(team_llr, live_ll, tau_team, tau_live):
    e = np.zeros((len(team_llr), 3))
    e[:, H] = live_ll[:, 0] / tau_live + 0.5 * team_llr / tau_team
    e[:, A] = live_ll[:, 0] / tau_live - 0.5 * team_llr / tau_team
    e[:, D] = live_ll[:, 1] / tau_live
    return e


def forward_backward(e, logT, log_pi):
    n = len(e); la = np.empty((n, 3)); lb = np.zeros((n, 3))
    la[0] = log_pi + e[0]
    for t in range(1, n):
        la[t] = e[t] + logsumexp(la[t - 1][:, None] + logT, axis=0)
    for t in range(n - 2, -1, -1):
        lb[t] = logsumexp(logT + (e[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = logsumexp(la[-1])
    gamma = np.exp(la + lb - ll)
    # expected transition counts, vectorised over time
    xi = np.exp(la[:-1, :, None] + logT[None] + (e[1:] + lb[1:])[:, None, :] - ll).sum(0)
    return gamma, xi, ll


def viterbi(e, logT, log_pi):
    n = len(e); delta = log_pi + e[0]; back = np.empty((n, 3), np.int8)
    for t in range(1, n):
        s = delta[:, None] + logT
        back[t] = np.argmax(s, axis=0); delta = s[back[t], range(3)] + e[t]
    path = np.empty(n, np.int8); path[-1] = np.argmax(delta)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def tie_transitions(xi_sum):
    """Symmetric H/A structure: one rate for H<->A, one for alive->D, one for D->alive (split by
    nothing -- D->H and D->A share a rate, the emissions decide which)."""
    c = xi_sum
    stay_alive = c[H, H] + c[A, A]; ha = c[H, A] + c[A, H]; ad = c[H, D] + c[A, D]
    tot_alive = stay_alive + ha + ad
    p_ha, p_ad = ha / tot_alive, ad / tot_alive
    tot_dead = c[D, D] + c[D, H] + c[D, A]; p_da = (c[D, H] + c[D, A]) / tot_dead
    T = np.array([[1 - p_ha - p_ad, p_ha, p_ad],
                  [p_ha, 1 - p_ha - p_ad, p_ad],
                  [p_da / 2, p_da / 2, 1 - p_da]])
    return T


def rescale(T, k):
    """Transition matrix for a k-times finer time step (first-order in the off-diagonal rates)."""
    off = T / k; np.fill_diagonal(off, 0); np.fill_diagonal(off, 1 - off.sum(1)); return off
