# Improving speed with polyominoes sampling.

Key idea: To further reduce the number of collages needed for object detection (reduce the number of object detection calls), we remove some polyominoes before the bin-packing step.
How to choose which polyominoes to remove? Given a maximum frame sampling gap for each tile position, our system chooses a set of polyominoes to pack such that the sampling gaps for each tile position does not exceed its maximum frame sampling gap. For example, the tile at position (i, j) has a maximum frame sampling gap of ti,j. Then, all the consecutive polyominoes chosen that contain the tile at the position (i, j) should not be more than ti,j frames apart from each other.

To optimize for fewer polyominoes (and fewer collages), we can follow 2 approaches.
1. Greedy for least amount of polyominoes.
2. NP for least amount of tiles.

To define the problems more formally, we see a video with B frames as a sequence of B chess boards of size N x M tiles.
Tile(b, n, m) denotes a tile at board b at the position (n, m), where 0<=b<B, 0<=n<N, 0<=m<M.
Each tile position (n, m), where 0<=n<N,0<=m<M has a maximum sampling gap of tn,m. This means that if a tile at (n, m) is chosen at the board b, the tile at (n, m) must be chosen again before or at the board b+t_{n,m}. Note that there is one t_{n,m} number per position, meaning that all Tile(*, n, m) are enforced to the same t_{n,m}. We can see this number as a timer. If a Tile(b, n, m) is chosen, a timer starts at t_{n,m} and decreases by 1 as the board progresses. Another tile at (n, m) must be chosen again before the timer runs out.
Each Tile(b, n, m) may be positive (relevant) or negative (irrelevant). N/E/W/S-connected positive tiles within the same board b form a positive polyomino. If one positive tile is selected, all of the tiles from its positive polyomino need to be selected too.
If a Tile(b, n, m) is positive, but there is no other positive tile within b+1, then, Tile(b, n, m) must be chosen and the next earliest positive Tile(b', n, m) must be chosen. (Impossible Covering)

Outlining 2 strategies for selecting positive polyominoes.
1. Greedy Temporal Covering Algorithm: Selecting the least amount of polyominoes.
  - Fast
  - Not necessarily the least amount of tiles being selected.
2. Integer Linear Programming (ILP): Selecting the least amount of tiles.
  - NP

## 1. Greedy Temporal Covering Algorithm
Objective: Minimize the total number of selection events (frequency of picks), not optimal.
### High-Level Idea
This algorithm uses a lazy approach. It tracks the specific deadline for every tile currently "active" in the system. It only performs a selection when a tile is at the absolute limit of its timer. When a selection is forced, it chooses the latest possible board to push the next deadline as far into the future as possible.
### Pseudocode (Text)
1. Identify all connected components (positive polyominoes) on all boards.
2. Initialize Deadlines for all tiles based on Board 0 mandatory selections.
3. While there are pending deadlines:
  - Find cell (n,m) with the earliest deadline D.
  - If the next positive instance of (n,m) is at board b_{next}​>D:
    - Select the component at the latest board b≤D.
    - Select the component at b_{next}​ (Constraint 4).
  - Else:
    - Select the component containing (n,m) at the latest board b≤D.
  - Update deadlines for all cells in the selected components to b_{selected}​+t_{n,m}​.

### Pseudocode (Python)
```python
def greedy_scheduler(B, N, M, boards, timers, components):
    # components[b][n][m] maps cell to its CC_ID
    # cc_data[b][cc_id] returns list of (n, m) in that CC
    last_selected = { (n,m): 0 for n,m in positive_cells_at_0 }
    pq = [(0 + timers[n][m], (n,m)) for n,m in last_selected]
    heapq.heapify(pq)
    
    selected_ccs = set([(0, cid) for cid in all_cids_at_0])

    while pq:
        deadline, (n, m) = heapq.heappop(pq)
        if last_selected[n,m] >= deadline - timers[n,m]: continue
        
        # Find latest board b <= deadline where (n,m) is positive
        target_b = find_latest(n, m, end=deadline)
        
        # Constraint 4 logic
        if target_b is None:
            target_b = find_earliest(n, m, start=deadline)
        
        cid = components[target_b][n,m]
        selected_ccs.add((target_b, cid))
        
        for (cn, cm) in cc_data[target_b][cid]:
            last_selected[cn, cm] = target_b
            heapq.heappush(pq, (target_b + timers[cn, cm], (cn, cm)))
```
### Time Complexity
O(BNM log(NM)). The complexity is dominated by the priority queue operations. Pre-processing components takes linear time O(BNM).

## 2. Integer Linear Programming (ILP)
Objective: Minimize the total number of tiles selected using a constrained optimization solver.
### High-Level Idea
We define binary decision variables for every polyomino. Constraints are formulated as linear inequalities that ensure every tile (n,m) is "covered" at least once in every tn,m​ window. Impossible Covering Constraint is handled by pre-fixing specific variables to 1.
### Pseudocode (Text)
1. Variables: x_{b,k}​∈{{0,1}} for component k on board b.
2. Objective: Minimize ∑x_{b,k}​size(C_{b,k}​).
3. Temporal Constraint: For every cell (n,m) and every positive instance at b_{curr}​, define window W={{ b ∈ Pos_{n,m}​ ∣ b_{curr}​ < b ≤ b_{curr}​ + t_{n,m​} }}.
  - If W is not empty: ∑_{b∈W}​x_{b,comp} ​≥ x_{b_{curr}​,comp​}.
  - If W is empty (Impossible Covering Constraint): x_{b_{curr}​,comp}​=1 AND x_{b_{next}​,comp}​=1.
4. Boundary: Force x_{0,k​}=1 and x_{B−1,k​}=1.
### Pseudocode (Python)
```python
import pulp

def solve_ilp(B, N, M, boards, timers, components):
    prob = pulp.LpProblem("MinCells", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", [(b, k) for b in range(B) 
                                   for k in range(len(components[b]))], cat='Binary')

    # Objective
    prob += pulp.lpSum([x[b,k] * len(components[b][k]) for b,k in x])

    # Constraint 4 & Temporal Logic
    for n, m in all_cells:
        pos = [b for b in range(B) if boards[b][n][m] == 1]
        for i, b_curr in enumerate(pos[:-1]):
            b_next_avail = pos[i+1]
            t_limit = b_curr + timers[n][m]
            
            curr_var = x[(b_curr, cell_to_cid[b_curr][n,m])]
            
            if b_next_avail > t_limit:
                # Constraint 4: Mandatory bridge
                prob += curr_var == 1
                prob += x[(b_next_avail, cell_to_cid[b_next_avail][n,m])] == 1
            else:
                # Window constraint
                window = [b for b in pos[i+1:] if b <= t_limit]
                prob += pulp.lpSum([x[(b, cell_to_cid[b][n,m])] for b in window]) >= curr_var

    prob.solve()
    return prob
```
TODO: If a tile (n, m) is always negative → remove the variable.