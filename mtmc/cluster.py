"""Agrupamento hierárquico (ligação média) de tracklets com restrições.

Nós = tracklets; arestas conhecidas = pares candidatos com pontuação (logit). Funde-se o par de grupos de maior ligação média enquanto
  * a ligação média (só entre pares conhecidos) >= `tau`;
  * nenhuma aresta conhecida entre os grupos é pior que `veto` (veto duro: um par claramente diferente impede a fusão);
  * os grupos não têm duas observações da mesma câmera (um veículo não está duas vezes na mesma câmera).
Pares desconhecidos (fora da janela temporal) não contam a favor nem contra.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Sequence, Tuple


def agglomerate(n: int, cams: Sequence[str], edges: Dict[Tuple[int, int], float], tau: float = 0.0, veto: float = -4.0) -> List[List[int]]:
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    camset: Dict[int, set] = {i: {cams[i]} for i in range(n)}
    stats: Dict[int, Dict[int, List[float]]] = {i: {} for i in range(n)}      # grupo -> vizinho -> [soma, n, mínimo]
    for (i, j), s in edges.items():
        stats[i][j] = stats[j][i] = [s, 1, s]
    ver = {i: 0 for i in range(n)}
    heap: List[Tuple[float, int, int, int, int]] = []

    def push(i: int, j: int):
        s, c, _ = stats[i][j]
        heapq.heappush(heap, (-(s / c), i, j, ver[i], ver[j]))

    for i in range(n):
        for j in stats[i]:
            if i < j:
                push(i, j)
    nxt = n
    while heap:
        neg, i, j, vi, vj = heapq.heappop(heap)
        if i not in members or j not in members or ver[i] != vi or ver[j] != vj:
            continue
        avg = -neg
        if avg < tau:
            break                                                          # fila ordenada: o resto é pior
        s, c, mn = stats[i][j]
        if mn < veto or camset[i] & camset[j]:
            continue                                                       # bloqueado (veto/câmera repetida)
        k = nxt
        nxt += 1
        members[k] = members.pop(i) + members.pop(j)
        camset[k] = camset.pop(i) | camset.pop(j)
        stats[k] = {}
        ver[k] = 0
        for old in (i, j):
            for nb, (ss, cc, mm) in stats.pop(old).items():
                if nb in (i, j):
                    continue
                stats[nb].pop(old, None)
                cur = stats[k].get(nb)
                stats[k][nb] = [ss, cc, mm] if cur is None else [cur[0] + ss, cur[1] + cc, min(cur[2], mm)]
        for nb, v in stats[k].items():
            stats[nb][k] = v
            push(k, nb)
    return list(members.values())
