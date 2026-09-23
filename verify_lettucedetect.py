"""Sanity-check that LettuceDetect really detects hallucinations.

Setup:
    pip install lettucedetect -U
Run:
    python verify_lettucedetect.py

Each test case has a known ground truth. A working detector must:
  1. flag nothing on faithful answers (no false positives),
  2. flag the injected fact on hallucinated answers (true positives),
  3. change its verdict when ONLY the context changes (proves it reads the context,
     not just memorised world knowledge).
"""

from lettucedetect.models.inference import HallucinationDetector

MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

CASES = [
    # (name, context, question, answer, expected_hallucinated_substring or None)
    (
        "faithful answer",
        ["France is in Europe. The capital of France is Paris. Its population is 67 million."],
        "What is the capital and population of France?",
        "The capital of France is Paris and its population is 67 million.",
        None,
    ),
    (
        "wrong number",
        ["France is in Europe. The capital of France is Paris. Its population is 67 million."],
        "What is the capital and population of France?",
        "The capital of France is Paris and its population is 69 million.",
        "69 million",
    ),
    (
        "invented fact",
        ["The Eiffel Tower was completed in 1889 and is 330 metres tall."],
        "Tell me about the Eiffel Tower.",
        "The Eiffel Tower was completed in 1889, is 330 metres tall and was designed by Leonardo da Vinci.",
        "Leonardo da Vinci",
    ),
    # Counterfactual pair: same answer, context flipped. A real grounding detector
    # must follow the context, even when the context contradicts world knowledge.
    (
        "counterfactual: context says Lyon",
        ["In this fictional world, the capital of France is Lyon."],
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Paris",
    ),
    (
        "counterfactual: context says Paris",
        ["The capital of France is Paris."],
        "What is the capital of France?",
        "The capital of France is Paris.",
        None,
    ),
]


def main() -> None:
    detector = HallucinationDetector(method="transformer", model_path=MODEL)
    passed = 0
    for name, context, question, answer, expected in CASES:
        spans = detector.predict(
            context=context, question=question, answer=answer, output_format="spans"
        )
        flagged = " | ".join(f"{s['text'].strip()!r} ({s['confidence']:.2f})" for s in spans)
        if expected is None:
            ok = not spans
        else:
            ok = any(expected in s["text"] for s in spans)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"    expected: {expected!r}   flagged: {flagged or 'nothing'}")

    print(f"\n{passed}/{len(CASES)} checks passed")

    # Token-level view: see the raw per-token probability the model assigns.
    print("\nToken probabilities for the 'wrong number' case:")
    _, ctx, q, ans, _ = CASES[1]
    for t in detector.predict(context=ctx, question=q, answer=ans, output_format="tokens"):
        print(f"    {t['token']!r:>14}  p(halluc)={t['prob']:.3f}  pred={t['pred']}")


if __name__ == "__main__":
    main()
