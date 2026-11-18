"""
HMM + Viterbi Route Reconstruction for PathSnap.

Hidden states:  candidate road arcs for each GPS observation.
Emissions:      Random Forest predicted P(on_route=1) per candidate.
Transitions:    road-network connectivity + network distance + GPS displacement/time.

All probabilities work in log-space to avoid underflow on long sequences.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6371008.8


def _project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres."""
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.c_[x, y]


class HMMRouteReconstructor:
    """
    Encapsulates the road network and Viterbi decoding for one trip.

    Parameters
    ----------
    nodes : np.ndarray (N, 2)   lon, lat per node (row index = node id)
    arcs  : np.ndarray (A, 2)  start_node, end_node per arc (row index = arc id)
    """

    def __init__(self, nodes, arcs):
        self.nodes = nodes
        self.arcs = arcs.astype(int)
        self.n_arcs = len(arcs)
        self.n_nodes = len(nodes)
        lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
        self.lon0, self.lat0 = lon0, lat0
        self.node_xy = _project(nodes[:, 0], nodes[:, 1], lon0, lat0)
        # arc geometry
        u, v = self.arcs[:, 0], self.arcs[:, 1]
        self.arc_start = self.node_xy[u]
        self.arc_end = self.node_xy[v]
        self.arc_len = np.linalg.norm(self.arc_end - self.arc_start, axis=1)
        self.arc_bearing = np.arctan2(self.arc_end[:, 1] - self.arc_start[:, 1],
                                       self.arc_end[:, 0] - self.arc_start[:, 0])
        # adjacency: which arcs start at each node
        self.out_arcs = {}
        for a in range(self.n_arcs):
            end_node = self.arcs[a, 1]
            self.out_arcs.setdefault(end_node, []).append(a)
        # canon_id for duplicate arcs
        canon = pd.Series(np.arange(self.n_arcs)).groupby([u, v]).transform("min").to_numpy()
        self.canon_id = canon

    # ---- emission ----------------------------------------------------------
    @staticmethod
    def emission_log_prob(proba, eps=1e-12):
        """Safe log of P(on_route=1) from the classifier."""
        return np.log(np.clip(proba, eps, 1.0 - eps))

    # ---- transition --------------------------------------------------------
    def transition_log_prob(self, arc_prev, arc_cur, gps_dist, dt,
                            beta_d=0.06, beta_t=0.1, not_connected_penalty=15.0):
        """
        Transition score between consecutive candidate arcs.

        Components (all additive in log-space):
        1. Connectivity bonus:  arc_prev end-node == arc_cur start-node.
        2. Same-arc bonus:      consecutive points stay on the same arc.
        3. Network distance:   shortest road-path estimate between arc ends.
        4. GPS displacement:   penalise transitions where the implied road
                               distance is very different from the GPS displacement.

        Parameters
        ----------
        arc_prev, arc_cur : int   arc IDs of the previous and current candidate.
        gps_dist : float            GPS displacement between the two points (metres).
        dt : float                  time gap (seconds).
        beta_d : float              decay rate for distance mismatch.
        beta_t : float              decay rate for time gap.
        not_connected_penalty : float   log-penalty when arcs are not connected.

        Returns
        -------
        float   log transition score (higher = better).
        """
        # same arc
        if arc_prev == arc_cur or self.canon_id[arc_prev] == self.canon_id[arc_cur]:
            return 0.0  # neutral: staying on the same road

        end_prev = self.arcs[arc_prev, 1]   # end node of previous arc
        start_cur = self.arcs[arc_cur, 0]   # start node of current arc

        # connectivity
        connected = (end_prev == start_cur)
        if connected:
            conn_score = 0.0
            net_dist = 0.0
        else:
            # estimate network distance as Euclidean between end of prev and start of cur
            p1 = self.arc_end[arc_prev]
            p2 = self.arc_start[arc_cur]
            net_dist = np.linalg.norm(p2 - p1)
            conn_score = -not_connected_penalty

        # distance mismatch penalty
        if gps_dist is not None and gps_dist > 0:
            # how well does network distance match GPS displacement?
            implied_dist = self.arc_len[arc_prev] + net_dist + self.arc_len[arc_cur] * 0.5
            ratio = max(implied_dist / max(gps_dist, 1.0), 1.0)
            dist_score = -beta_d * np.log(ratio)
        else:
            dist_score = 0.0

        # time gap penalty
        if dt is not None and dt > 0:
            time_score = -beta_t * np.log(max(dt, 1.0))
        else:
            time_score = 0.0

        return conn_score + dist_score + time_score

    # ---- Viterbi -----------------------------------------------------------
    def viterbi(self, candidate_lists, emission_probs, gps_xy, gps_dt):
        """
        Viterbi decoding for the most likely sequence of road arcs.

        Parameters
        ----------
        candidate_lists : list[list[int]]
            candidate_lists[t] = list of arc IDs for GPS observation t.
        emission_probs : list[np.ndarray]
            emission_probs[t] = P(on_route=1) for each candidate at time t.
        gps_xy : np.ndarray (T, 2)
            projected GPS coordinates (metres).
        gps_dt : np.ndarray (T,)
            time gap from the previous observation (seconds), 0 at segment start.

        Returns
        -------
        selected_arcs : list[int]
            one arc ID per observation (or -1 for observations with no candidates).
        """
        T = len(candidate_lists)
        if T == 0:
            return []

        # ---- initialise ----
        # delta[t, i] = best log-prob of being in state i at time t
        # psi[t, i]   = back-pointer to the best previous state
        deltas = []
        backptrs = []
        selected = [-1] * T

        # ---- emission for t=0 ----
        em0 = self.emission_log_prob(emission_probs[0])
        deltas.append(em0.copy())
        backptrs.append(np.zeros(len(candidate_lists[0]), dtype=int))

        # ---- forward pass ----
        for t in range(1, T):
            n_prev = len(candidate_lists[t - 1])
            n_cur = len(candidate_lists[t])
            if n_cur == 0:
                # no candidates for this observation
                deltas.append(np.array([]))
                backptrs.append(np.array([], dtype=int))
                continue

            # compute transition scores from every prev candidate to every cur candidate
            gps_dist = None
            if gps_xy is not None and t < len(gps_xy):
                d = np.linalg.norm(gps_xy[t] - gps_xy[t - 1])
                gps_dist = d if d > 0 else None
            dt_val = float(gps_dt[t]) if gps_dt is not None and t < len(gps_dt) else 0.0

            em_cur = self.emission_log_prob(emission_probs[t])
            delta_cur = np.full(n_cur, -np.inf)

            if n_prev == 0:
                # previous had no candidates → restart from emission
                delta_cur = em_cur.copy()
                backptrs.append(np.zeros(n_cur, dtype=int))
                deltas.append(delta_cur)
                continue

            # transition matrix (n_prev, n_cur)
            trans = np.zeros((n_prev, n_cur))
            for i in range(n_prev):
                a_prev = candidate_lists[t - 1][i]
                for j in range(n_cur):
                    a_cur = candidate_lists[t][j]
                    trans[i, j] = self.transition_log_prob(
                        a_prev, a_cur, gps_dist, dt_val)

            # Viterbi: delta_cur[j] = em_cur[j] + max_i(delta_prev[i] + trans[i,j])
            combined = deltas[t - 1][:, None] + trans  # (n_prev, n_cur)
            best_prev = np.argmax(combined, axis=0)     # (n_cur,)
            delta_cur = em_cur + combined[best_prev, np.arange(n_cur)]
            backptrs.append(best_prev)
            deltas.append(delta_cur)

        # ---- backward pass ----
        # find the last observation with candidates
        last_valid = T - 1
        while last_valid >= 0 and len(candidate_lists[last_valid]) == 0:
            last_valid -= 1
        if last_valid < 0:
            return selected  # all empty

        # start from best last state
        best_j = int(np.argmax(deltas[last_valid]))
        selected[last_valid] = candidate_lists[last_valid][best_j]

        # trace back
        for t in range(last_valid, 0, -1):
            if len(candidate_lists[t]) == 0 or len(candidate_lists[t - 1]) == 0:
                continue
            best_j = backptrs[t][best_j]
            selected[t - 1] = candidate_lists[t - 1][best_j]

        return selected

    # ---- convenience --------------------------------------------------------
    def reconstruct(self, candidate_lists, emission_probs, gps_xy, gps_dt):
        """Full pipeline: Viterbi decode and return selected arc per observation."""
        return self.viterbi(candidate_lists, emission_probs, gps_xy, gps_dt)


def reconstruct_route(net, x, y, t):
    """
    Backend-compatible interface (matches main.py's MODEL_ENTRY contract).
    net is a dict from backend/main.py's load_network.
    """
    # This is a placeholder for backend integration; the evaluation script
    # uses HMMRouteReconstructor directly with the raw .nodes/.arcs files.
    raise NotImplementedError("Use HMMRouteReconstructor directly for evaluation.")
