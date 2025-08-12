import graphviz
from teacher_graph import Orchestrator

def visualize_teacher_graph():
    # 오케스트레이터 생성 (더미 값으로 user_id/service/chat_id)
    orch = Orchestrator(user_id="demo", service="teacher", chat_id="test")
    compiled_graph = orch.build_teacher_graph()

    # LangGraph의 내부 graph 객체 가져오기
    g = compiled_graph.get_graph()

    dot = graphviz.Digraph(comment="Teacher Graph", format="png")
    dot.attr(rankdir="LR")  # 좌→우 방향

    # 노드 추가
    for node in g.nodes:
        dot.node(node, shape="box")

    # 엣지 추가
    for edge in g.edges:
        dot.edge(edge[0], edge[1])

    dot.render("teacher_graph", cleanup=True)
    print("그래프 이미지 저장됨: teacher_graph.png")

if __name__ == "__main__":
    visualize_teacher_graph()
