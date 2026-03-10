import random


def cascade_step_capacity(G, overload_factor=0.8):
    """
    Cascading model with load redistribution from failed roads.
    This produces gradual but real cascading waves.
    """

    new_failures = []

    # collect failed roads
    failed_nodes = [n for n, d in G.nodes(data=True) if d.get("failed")]

    if not failed_nodes:
        return []

    for n, d in G.nodes(data=True):

        if d.get("type") != "road" or d.get("failed"):
            continue

        base = d.get("base_traffic", d.get("traffic", 0))
        capacity = d.get("capacity", 1)

        # count failed neighbors
        failed_neighbors = 0

        for nbr in G.neighbors(n):
            nd = G.nodes[nbr]
            if nd.get("type") == "road" and nd.get("failed"):
                failed_neighbors += 1

        if failed_neighbors == 0:
            continue

        # load redistribution: each failed road pushes load outward
        overload = base + overload_factor * base * failed_neighbors

        # accumulate stress
        prev_stress = d.get("stress", 0)
        stress = prev_stress + overload / capacity
        d["stress"] = stress

        # probabilistic failure
        if stress > 1.0:
            prob = min(1.0, (stress - 1.0) * 1.5)

            if random.random() < prob:
                d["failed"] = True
                new_failures.append(n)

    return new_failures