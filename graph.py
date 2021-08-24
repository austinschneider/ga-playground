# +
# Bron Kerbosch implementation
# Obtained from https://github.com/ssomers/Bron-Kerbosch
# Licensed under BSD-3-Clause

from abc import abstractmethod, ABCMeta
from typing import Callable, List, Set

Vertex = int
PivotChoice = Callable

class Reporter(metaclass=ABCMeta):
    @abstractmethod
    def record(self, clique):
        pass

class SimpleReporter(Reporter):
    def __init__(self):
        self.cliques = []

    def record(self, clique):
        assert len(clique) > 1
        self.cliques.append(clique)


class CountingReporter(Reporter):
    def __init__(self):
        self.cliques = 0

    def record(self, clique):
        self.cliques += 1

class UndirectedGraph(object):
    def __init__(self, adjacencies: List[Set[Vertex]]):
        order = len(adjacencies)
        for v, adjacent_to_v in enumerate(adjacencies):
            for w in adjacent_to_v:
                assert 0 <= w < order
                assert v != w
                assert v in adjacencies[w], (
                    f'{w} is adjacent to {v} but not vice versa')
        self.adjacencies = adjacencies

    @property
    def order(self) -> int:
        return len(self.adjacencies)

    def size(self) -> int:
        total = sum(len(a) for a in self.adjacencies)
        assert total % 2 == 0
        return total // 2

    def degree(self, node):
        return len(self.adjacencies[node])

    def connected_vertices(self) -> Set[Vertex]:
        return {node for node in range(self.order) if self.adjacencies[node]}

def visit(graph: UndirectedGraph, reporter: Reporter,
          initial_pivot_choice: PivotChoice, further_pivot_choice: PivotChoice,
          candidates: Set[Vertex], excluded: Set[Vertex],
          clique: List[Vertex]):
    assert candidates
    assert all(graph.degree(v) > 0 for v in candidates)
    assert all(graph.degree(v) > 0 for v in excluded)

    if len(candidates) == 1:
        # Same logic as below, stripped down for this common case
        for v in candidates:
            neighbours = graph.adjacencies[v]
            assert neighbours
            if excluded.isdisjoint(neighbours):
                reporter.record(clique + [v])
        return

    if initial_pivot_choice in [pick_max_degree_local, pick_max_degree_localX]:
        # Quickly handle locally unconnected candidates while finding pivot
        remaining_candidates = []
        seen_local_degree = 0
        for v in candidates:
            neighbours = graph.adjacencies[v]
            local_degree = len(neighbours.intersection(candidates))
            if local_degree == 0:
                # Same logic as below, stripped down
                if neighbours.isdisjoint(excluded):
                    reporter.record(clique + [v])
            else:
                if seen_local_degree < local_degree:
                    seen_local_degree = local_degree
                    pivot = v
                remaining_candidates.append(v)
        if seen_local_degree == 0:
            return
        if initial_pivot_choice == pick_max_degree_localX:
            for v in excluded:
                neighbours = graph.adjacencies[v]
                local_degree = len(neighbours.intersection(candidates))
                if seen_local_degree < local_degree:
                    seen_local_degree = local_degree
                    pivot = v
    else:
        pivot = initial_pivot_choice(graph=graph, candidates=candidates)
        remaining_candidates = list(candidates)

    for v in remaining_candidates:
        neighbours = graph.adjacencies[v]
        assert neighbours
        if pivot in neighbours:
            continue
        candidates.remove(v)
        if neighbouring_candidates := candidates.intersection(neighbours):
            neighbouring_excluded = excluded.intersection(neighbours)
            visit(graph=graph,
                  reporter=reporter,
                  initial_pivot_choice=further_pivot_choice,
                  further_pivot_choice=further_pivot_choice,
                  candidates=neighbouring_candidates,
                  excluded=neighbouring_excluded,
                  clique=clique + [v])
        else:
            if excluded.isdisjoint(neighbours):
                reporter.record(clique + [v])
        excluded.add(v)

def pick_max_degree(graph: UndirectedGraph, candidates: Set[Vertex]) -> Vertex:
    return max(candidates, key=graph.degree)

def pick_max_degree_local():
    pass


def pick_max_degree_localX():
    pass

class PriorityQueue:
    def __init__(self, max_priority):
        self.stack_per_priority = [[] for _ in range(max_priority + 1)]

    def put(self, priority, element):
        assert priority >= 0
        self.stack_per_priority[priority].append(element)

    def pop(self):
        for stack in self.stack_per_priority:
            try:
                return stack.pop()
            except IndexError:
                pass

def degeneracy_ordering(graph: UndirectedGraph, drop=0):
    """
    Iterate connected vertices, lowest degree first.
    drop=N: omit last N vertices
    """
    assert drop >= 0
    priority_per_node = [-2] * graph.order
    max_degree = 0
    num_candidates = 0
    for c in range(graph.order):
        if degree := graph.degree(c):
            priority_per_node[c] = degree
            max_degree = max(max_degree, degree)
            num_candidates += 1
    # Possible values of priority_per_node:
    #   -2: if unconnected (should never come up again)
    #   -1: when yielded
    #   0..max_degree: candidates still queued with priority (degree - #of yielded neighbours)
    q = PriorityQueue(max_priority=max_degree)
    for c, p in enumerate(priority_per_node):
        if p > 0:
            q.put(priority=p, element=c)

    for _ in range(num_candidates - drop):
        i = q.pop()
        while priority_per_node[i] == -1:
            # was requeued with a more urgent priority and therefore already picked
            i = q.pop()
        assert priority_per_node[i] >= 0
        priority_per_node[i] = -1
        yield i
        for v in graph.adjacencies[i]:
            if (p := priority_per_node[v]) != -1:
                assert p > 0
                # Requeue with a more urgent priority, but don't bother to remove
                # the original entry - it will be skipped if it's reached at all.
                priority_per_node[v] = p - 1
                q.put(priority=p - 1, element=v)

def bron_kerbosch3_gpx(graph: UndirectedGraph, reporter: Reporter):
    '''Bron-Kerbosch algorithm with degeneracy ordering, with nested searches
       choosing a pivot from both candidates and excluded vertices (IK_GPX)'''
    excluded: Set[Vertex] = set()
    for v in degeneracy_ordering(graph=graph, drop=1):
        neighbours = graph.adjacencies[v]
        assert neighbours
        if neighbouring_candidates := neighbours.difference(excluded):
            neighbouring_excluded = neighbours.intersection(excluded)
            visit(
                graph=graph,
                reporter=reporter,
                initial_pivot_choice=pick_max_degree_localX,
                further_pivot_choice=pick_max_degree_localX,
                candidates=neighbouring_candidates,
                excluded=neighbouring_excluded,
                clique=[v])
        else:
            assert not excluded.isdisjoint(neighbours)
        excluded.add(v)

