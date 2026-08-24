from __future__ import annotations


def _check(problem):
    if not isinstance(problem, list):
        raise ValueError('problem must be a list')
    n = len(problem)
    if any(not isinstance(r, list) or len(r) != n for r in problem):
        raise ValueError('adjacency matrix must be square')
    return n


def _masks(problem):
    n = _check(problem)
    out = [0] * n
    for i, row in enumerate(problem):
        m = 0
        for j, x in enumerate(row):
            if i != j and x:
                m |= 1 << j
        out[i] = m
    return out


def reproduced_bfr(problem):
    """Separate exact reimplementation inspired only by the abstract TM-BFR recipe.

    State is word-parallel; recursion operates only on the remaining candidate
    frontier; exactness is preserved by exhaustive include/exclude branching with
    safe cardinality pruning. Isolated frontier vertices are reduced eagerly.
    """
    n = _check(problem)
    if n == 0:
        return []
    adj = _masks(problem)
    universe = (1 << n) - 1

    def greedy_seed(avail):
        picked = 0
        while avail:
            scan = avail
            choice = -1
            degree = 10**18
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                d = (adj[v] & avail).bit_count()
                if d < degree:
                    choice, degree = v, d
                scan ^= bit
            bit = 1 << choice
            picked |= bit
            avail &= ~bit
            avail &= ~adj[choice]
        return picked

    incumbent = greedy_seed(universe)
    best_mask = incumbent
    best_size = incumbent.bit_count()

    def visit(avail, selected, selected_size):
        nonlocal best_mask, best_size

        # Eagerly include vertices isolated inside the current frontier.
        while avail:
            scan = avail
            isolated = 0
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                if not (adj[v] & avail):
                    isolated |= bit
                scan ^= bit
            if not isolated:
                break
            selected |= isolated
            selected_size += isolated.bit_count()
            avail &= ~isolated

        if selected_size + avail.bit_count() <= best_size:
            return
        if not avail:
            if selected_size > best_size:
                best_mask, best_size = selected, selected_size
            return

        scan = avail
        pivot = -1
        pivot_degree = -1
        while scan:
            bit = scan & -scan
            v = bit.bit_length() - 1
            d = (adj[v] & avail).bit_count()
            if d > pivot_degree:
                pivot, pivot_degree = v, d
            scan ^= bit
        bit = 1 << pivot

        # Include pivot: its neighbors leave the independent-set frontier.
        visit(avail & ~bit & ~adj[pivot], selected | bit, selected_size + 1)
        # Exclude pivot.
        visit(avail & ~bit, selected, selected_size)

    visit(universe, 0, 0)
    return [i for i in range(n) if not ((best_mask >> i) & 1)]


def color_bound_clique_cover(problem):
    """Known-style strong post-hoc baseline.

    Computes maximum independent set in G as maximum clique in complement(G),
    using a bitset branch-and-bound with greedy coloring upper bounds. This is
    intentionally *not* a v5 arm; it probes whether the benchmark SAT reference
    is weak relative to standard exact graph-search techniques.
    """
    n = _check(problem)
    if n == 0:
        return []
    adj = _masks(problem)
    all_bits = (1 << n) - 1
    comp = [0] * n
    for v in range(n):
        comp[v] = all_bits & ~(1 << v) & ~adj[v]

    best_mask = 0
    best_size = 0

    def color_sort(cand):
        order = []
        bounds = []
        uncolored = cand
        color = 0
        while uncolored:
            color += 1
            available = uncolored
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                uncolored &= ~bit
                available &= ~bit
                # Same-color vertices must be pairwise nonadjacent in complement.
                available &= ~comp[v]
        return order, bounds

    def expand(cand, clique_mask, clique_size):
        nonlocal best_mask, best_size
        if not cand:
            if clique_size > best_size:
                best_mask, best_size = clique_mask, clique_size
            return
        order, bounds = color_sort(cand)
        for idx in range(len(order) - 1, -1, -1):
            if clique_size + bounds[idx] <= best_size:
                return
            v = order[idx]
            bit = 1 << v
            if not (cand & bit):
                continue
            nxt = cand & comp[v]
            if nxt:
                expand(nxt, clique_mask | bit, clique_size + 1)
            elif clique_size + 1 > best_size:
                best_mask, best_size = clique_mask | bit, clique_size + 1
            cand &= ~bit

    expand(all_bits, 0, 0)
    return [i for i in range(n) if not ((best_mask >> i) & 1)]
