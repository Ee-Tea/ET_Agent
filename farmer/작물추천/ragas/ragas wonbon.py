# ragas_eval_amnesty.py
# ------------------------------------------------------------
# Standalone script to evaluate a small subset of the Amnesty QA
# dataset with RAGAS metrics using an Ollama LLM + embeddings.
#
# Requirements (install once in your environment):
#   pip install -U datasets ragas langchain langchain-community
#   # And make sure Ollama is running with a 'llama3' model available.
#   # e.g., `ollama run llama3` or `ollama pull llama3`
# ------------------------------------------------------------

from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings


def main() -> None:
    # 1) Prepare Dataset
    # Note: trust_remote_code may be required in future versions of `datasets`.
    amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
    # Take a tiny subset for a quick demo
    amnesty_subset = amnesty_qa["eval"].select(range(2))

    # 2) Initialize model (Ollama must be running and have 'llama3' available)
    llm = ChatOllama(model="llama3")
    embeddings = OllamaEmbeddings(model="llama3")

    # 3) Evaluate
    result = evaluate(
        amnesty_subset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    # 4) Print results
    print(result)


if __name__ == "__main__":
    main()
