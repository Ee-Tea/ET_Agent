# teacher/teacher_graph_visual.py
from .teacher_graph import Orchestrator
import graphviz

def _as_iter_nodes(g):
    nodes_attr = getattr(g, "nodes", None)
    if callable(nodes_attr):
        return nodes_attr()  # view
    return nodes_attr or []  # list or empty

def _as_iter_edges(g, need_keys=False):
    edges_attr = getattr(g, "edges", None)
    if callable(edges_attr):
        # networkx EdgeView
        if need_keys:
            try:
                return edges_attr(keys=True)
            except TypeError:
                return edges_attr()  # fallback
        return edges_attr()
    # list-like: normalize to iterator of tuples
    edges_list = edges_attr or []
    return edges_list

def visualize_teacher_graph():
    # 실행 없이 그래프 구조만 빌드하므로 init_agents=False
    orch = Orchestrator(user_id="demo", service="teacher", chat_id="viz", init_agents=False)
    compiled = orch.build_teacher_graph()
    g = compiled.get_graph()

    dot = graphviz.Digraph(comment="Teacher Graph", format="png")
    dot.attr(rankdir="TD")

    # 노드
    for n in _as_iter_nodes(g):
        dot.node(str(n), shape="box")

    # 멀티그래프 여부
    is_multi = False
    is_multi_attr = getattr(g, "is_multigraph", None)
    if callable(is_multi_attr):
        is_multi = bool(is_multi_attr())

    # 엣지
    if is_multi:
        for e in _as_iter_edges(g, need_keys=True):
            # e가 (u,v,key) 또는 (u,v)일 수 있음
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                u, v = e[0], e[1]
                dot.edge(str(u), str(v))
    else:
        for e in _as_iter_edges(g, need_keys=False):
            # e가 (u,v) 또는 (u,v,data)일 수 있음
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                u, v = e[0], e[1]
                dot.edge(str(u), str(v))

    path = dot.render("teacher_graph", cleanup=True)
    print(f"saved: {path}")

if __name__ == "__main__":
    visualize_teacher_graph()
