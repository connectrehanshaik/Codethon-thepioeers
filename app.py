import streamlit as st
import re
import math
from collections import Counter

st.set_page_config(
    page_title="Semantic Plagiarism Detection Agent",
    layout="wide"
)

# --- Pure Python Lightweight Semantic Vector Engine ---

def clean_and_tokenize(text):
    text = text.lower()
    return re.findall(r'\b[a-z0-9]+\b', text)

def split_into_sentences(text):
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 15]

def get_character_ngrams(text, n=3):
    clean = re.sub(r'\s+', ' ', text.lower().strip())
    if len(clean) < n:
        return [clean]
    return [clean[i:i+n] for i in range(len(clean) - n + 1)]

def compute_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([val**2 for val in vec1.values()])
    sum2 = sum([val**2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return (float(numerator) / denominator) if denominator else 0.0

def calculate_sentence_similarity(sent1, sent2):
    tokens1 = clean_and_tokenize(sent1)
    tokens2 = clean_and_tokenize(sent2)
    ngrams1 = get_character_ngrams(sent1, 3)
    ngrams2 = get_character_ngrams(sent2, 3)

    word_sim = compute_cosine(Counter(tokens1), Counter(tokens2))
    ngram_sim = compute_cosine(Counter(ngrams1), Counter(ngrams2))

    # 35% lexical overlap + 65% sub-word structural semantics
    return (0.35 * word_sim) + (0.65 * ngram_sim)

# --- Streamlit UI ---

st.title("Semantic Plagiarism Detection Agent")
st.caption("AI-powered academic integrity engine detecting structural rewrites, synonym swaps, and paraphrased theft.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Reference / Source Text")
    ref_text = st.text_area("Paste original or published content:", height=200, key="ref")

with col2:
    st.subheader("Suspect Text")
    sus_text = st.text_area("Paste candidate or submitted content:", height=200, key="sus")

st.sidebar.header("Agent Settings")
threshold = st.sidebar.slider("Paraphrase Detection Sensitivity", 0.30, 0.90, 0.55, 0.01)

st.sidebar.markdown("""
**Threshold Guide:**
* **> 0.70:** Heavy synonym swapping / direct paraphrase.
* **0.50 – 0.69:** Structural restructuring of ideas.
* **< 0.50:** Broad thematic alignment.
""")

if st.button("Run Semantic Audit", type="primary", use_container_width=True):
    if not ref_text.strip() or not sus_text.strip():
        st.error("Please paste content into both Reference and Suspect text areas to analyze.")
    else:
        with st.spinner("Analyzing semantic structure and calculating vector distances..."):
            sents_ref = split_into_sentences(ref_text)
            sents_sus = split_into_sentences(sus_text)

            if not sents_ref or not sents_sus:
                st.warning("One of the inputs is too brief to tokenize into sentences. Add at least 1-2 complete sentences.")
            else:
                flagged_cases = []
                total_suspect = len(sents_sus)
                flagged_count = 0

                for s_idx, sus_s in enumerate(sents_sus):
                    best_score = 0.0
                    best_ref_match = ""
                    for r_idx, ref_s in enumerate(sents_ref):
                        sim = calculate_sentence_similarity(sus_s, ref_s)
                        if sim > best_score:
                            best_score = sim
                            best_ref_match = ref_s

                    if best_score >= threshold:
                        flagged_count += 1
                        flagged_cases.append({
                            "suspect": sus_s,
                            "reference": best_ref_match,
                            "score": best_score
                        })

                plagiarism_percentage = (flagged_count / total_suspect) * 100
                avg_confidence = (sum(c["score"] for c in flagged_cases) / len(flagged_cases)) if flagged_cases else 0.0

                st.subheader("Audit Analytics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Plagiarism Index", f"{plagiarism_percentage:.1f}%")
                m2.metric("Flagged Sentences", f"{flagged_count} / {total_suspect}")
                m3.metric("Avg Semantic Match", f"{avg_confidence*100:.1f}%" if flagged_cases else "0.0%")

                if plagiarism_percentage >= 40:
                    m4.error("Integrity: High Risk")
                elif plagiarism_percentage >= 15:
                    m4.warning("Integrity: Moderate Risk")
                else:
                    m4.success("Integrity: Clean")

                st.divider()
                st.subheader("Paraphrase Matching Sections")
                if not flagged_cases:
                    st.success("No sections breached the semantic threshold.")
                else:
                    for i, case in enumerate(flagged_cases, 1):
                        with st.expander(f"Match #{i} — Semantic Alignment: {case['score']*100:.1f}%", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Original Source Passage:**")
                                st.info(case["reference"])
                            with c2:
                                st.markdown("**Paraphrased Variant:**")
                                st.error(case["suspect"])

                st.divider()
                st.subheader("Formal Audit Report")
                report = f"""==================================================
        SEMANTIC PLAGIARISM AUDIT REPORT
==================================================
Total Suspect Units Evaluated : {total_suspect}
Flagged Paraphrased Units    : {flagged_count}
Plagiarism Index              : {plagiarism_percentage:.2f}%
Average Match Score           : {avg_confidence*100:.2f}%
Sensitivity Threshold Applied : {threshold}
Final Audit Decision          : {"FLAGGED: UNORIGINAL RESTRUCTURING / THEFT" if plagiarism_percentage > 25 else "ACCEPTED: ORIGINAL CONTENT"}
==================================================
Flagged Paraphrase Matches:
"""
                for idx, c in enumerate(flagged_cases, 1):
                    report += f"\n- Match #{idx} ({c['score']*100:.1f}%):"
                    report += f"\n  Suspect: {c['suspect']}"
                    report += f"\n  Source : {c['reference']}\n"

                st.text_area("Audit Log Output", report, height=160)
                st.download_button("Download Audit Report (.txt)", report, file_name="semantic_audit_report.txt")
