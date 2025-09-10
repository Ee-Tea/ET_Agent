# visualize_flowchart.py (v1.1: 1_create_golden_set.py 로직 반영)

from graphviz import Digraph
import os

def create_workflow_visualization():
    """
    '1_create_golden_set.py' 스크립트의 작동 순서도를 생성하고
    .png 파일로 저장하는 함수입니다.
    """
    # 그래프 객체 생성 (방향 그래프)
    dot = Digraph('GoldenSetWorkflow', comment='Golden Set Creation Workflow')
    dot.attr(rankdir='TB', label='1_create_golden_set.py 작동 순서도', fontsize='20', fontname='Malgun Gothic')
    dot.attr('node', shape='box', style='rounded', fontname='Malgun Gothic', fontsize='12')
    dot.attr('edge', fontname='Malgun Gothic', fontsize='10')

    # 노드(단계) 정의
    dot.node('start', '시작', shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('read_pdfs', 'PDF 폴더에서 모든 파일 읽기\n(PDF_INPUT_DIR)')
    dot.node('split_chunks', '각 PDF를 텍스트 조각(Chunk)으로 분할')
    dot.node('sample_chunks', '지정된 개수(N개)만큼 청크 샘플링\n(NUM_QUESTIONS_TO_GENERATE)')
    
    # 반복 작업을 표현하기 위한 서브그래프(클러스터)
    with dot.subgraph(name='cluster_loop') as c:
        c.attr(label='N번 반복', style='dashed')
        c.node('gen_question', 'LLM을 호출하여\n컨텍스트 기반 질문 생성')
        c.node('gen_gt_answer', 'LLM을 호출하여\n질문과 컨텍스트 기반 모범 답안 생성')
        c.node('collect_items', '생성된 질문과 모범 답안을\n리스트에 추가')
        c.edge('gen_question', 'gen_gt_answer')
        c.edge('gen_gt_answer', 'collect_items')

    dot.node('create_df', '수집된 질문-답변 쌍으로\nDataFrame 생성')
    dot.node('save_csv', '최종 결과를 CSV 파일로 저장\n(1_golden_set_...csv)')
    dot.node('end', '종료', shape='ellipse', style='filled', fillcolor='lightblue')

    # 엣지(흐름) 연결
    dot.edge('start', 'read_pdfs')
    dot.edge('read_pdfs', 'split_chunks')
    dot.edge('split_chunks', 'sample_chunks')
    dot.edge('sample_chunks', 'gen_question', lhead='cluster_loop')
    dot.edge('collect_items', 'create_df', ltail='cluster_loop')
    dot.edge('create_df', 'save_csv')
    dot.edge('save_csv', 'end')

    # 파일 렌더링 및 저장
    output_filename = '1_create_golden_set_workflow'
    try:
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"✅ 순서도가 '{output_filename}.png' 파일로 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("Graphviz가 시스템에 설치되어 있고 PATH가 올바르게 설정되었는지 확인하세요.")

if __name__ == "__main__":
    create_workflow_visualization()