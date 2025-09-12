# ragas_visualize.py
# -*- coding: utf-8 -*-
"""
RAGAS 평가 결과(JSON) 시각화 스크립트
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== 여기에 직접 파일 경로만 수정하면 됨 ====
FILE_PATH = "farmer/disaster/data/disaster_ragas_evaluation_results_20250911_165931.json"
# ============================================

def load_ragas_results(file_path: str) -> pd.DataFrame:
    """RAGAS JSON 결과를 DataFrame으로 변환"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON 구조: {"evaluation_summary": {...}, "detailed_results": [...]}
    if "detailed_results" in data:
        df = pd.DataFrame(data["detailed_results"])
        
        # individual_ragas_score를 별도 컬럼으로 분리
        if "individual_ragas_score" in df.columns:
            ragas_scores = df["individual_ragas_score"].apply(pd.Series)
            df = pd.concat([df.drop("individual_ragas_score", axis=1), ragas_scores], axis=1)
        
        return df
    else:
        # 기존 방식 (리스트 형태)
        df = pd.DataFrame(data)
        return df

def summary_stats(df: pd.DataFrame, metrics: list):
    """각 지표별 기본 통계"""
    return df[metrics].describe(percentiles=[0.25, 0.5, 0.75])

def visualize(df: pd.DataFrame, metrics: list):
    """히스토그램, 박스플롯, heatmap 그리기"""
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 실행
    
    # 1) 분포도 (히스토그램 + KDE + 점 플롯)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("RAGAS Metrics Distribution Analysis", fontsize=16)
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # 히스토그램
        ax.hist(df[metric], bins=15, alpha=0.3, color='skyblue', density=True, label='Histogram')
        
        # KDE (커널 밀도 추정)
        from scipy import stats
        kde = stats.gaussian_kde(df[metric])
        x_range = np.linspace(df[metric].min(), df[metric].max(), 100)
        ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
        
        # 점 플롯 (jitter 추가)
        y_jitter = np.random.normal(0, 0.01, len(df[metric]))
        ax.scatter(df[metric], y_jitter, alpha=0.4, s=30, color='darkblue', label='Data Points')
        
        # 통계 정보 추가
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle=':', alpha=0.8, label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(f'{metric} Distribution\n(Std: {std_val:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ragas_histogram.png", dpi=300, bbox_inches='tight')
    print("📊 분포도 저장: ragas_histogram.png")
    plt.close()

    # 2) 박스플롯
    plt.figure(figsize=(12, 6))
    df[metrics].plot(kind="box")
    plt.title("RAGAS Metrics Boxplot")
    plt.savefig("ragas_boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 박스플롯 저장: ragas_boxplot.png")
    plt.close()

    # 3) 상관관계 히트맵
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig("ragas_correlation.png", dpi=300, bbox_inches='tight')
    print("📊 상관관계 히트맵 저장: ragas_correlation.png")
    plt.close()

def find_worst_cases(df: pd.DataFrame, metric: str, n: int = 10):
    """특정 metric 기준으로 최악의 질문 출력"""
    worst = df.nsmallest(n, metric)[["question", metric]]
    print(f"\n📉 {metric} 최저 Top-{n} 질문")
    print(worst.to_string(index=False))

if __name__ == "__main__":
    df = load_ragas_results(FILE_PATH)

    # 평가 지표 (파일 구조에 맞게 조정 가능)
    metrics = ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]
    available_metrics = [m for m in metrics if m in df.columns]

    print("\n📊 기본 통계")
    print(summary_stats(df, available_metrics))

    visualize(df, available_metrics)

    for metric in available_metrics:
        find_worst_cases(df, metric, n=10)
