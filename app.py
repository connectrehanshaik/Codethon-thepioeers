import streamlit as st
import numpy as np
import re
import math
from collections import Counter

# Set page config
st.set_page_config(page_title="Plagiarism Detection System", layout="wide")

st.title("Semantic Plagiarism Detection Tool")
st.write("Detects paraphrased and restructured sentences using sub-word n-gram vector matching.")

# Side bar settings
cutoff = st.sidebar.slider("Similarity Cutoff", min_value=0.40, max_value=0.90, value=0.60, step=0.05)

# Input text areas
col1, col2 = st.columns(2)
with col1:
    orig_text = st.text_area("Reference Document (Original)", height=220, placeholder="Paste original text here...")
with col2:
    sus_text = st.text_area("Suspect Document (Submitted)", height=220, placeholder="Paste submitted text here...")


# --- text processing helpers ---

def tokenize(s):
    # strip symbols, just keep words & numbers
    return re.findall(r'[a-z0-9]+', s.lower())

def make_sentences(text):
    # split on punctuation dots, ?, !
    raw = re.split(r'[\.\?\!\n]+', text)
    # ignore short fragments
    clean = [item.strip() for item in raw if len(item.strip().split()) > 3]
    return clean

def get_ngrams(text, n=3):
    text = re.sub(r'\s+', ' ', text.lower().strip())
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def cos_similarity(vec1, vec2):
    # standard cosine similarity formula: dot(a, b) / (norm(a) * norm(b))
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[k] * vec2[k] for k in common)
    
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def calc_score(s1, s2):
    # blend word-level overlap and character 3-grams
    # n-grams catch synonym/stemming tricks
    w1, w2 = tokenize(s1), tokenize(s2)
    ng1, ng2 = get_ngrams(s1, 3), get_ngrams(s2, 3)
    
    word_sim = cos_similarity(Counter(w1), Counter(w2))
    ngram_sim = cos_similarity(Counter(ng1), Counter(ng2))
    
    final_score = (0.35 * word_sim) + (0.65 * ngram_sim)
    return round(final_score, 4)


# --- main logic ---

if st.button("Run Plagiarism Check", type="primary"):
    if not orig_text.strip() or not sus_text.strip():
        st.warning("Please provide input for both documents.")
    else:
        orig_sents = make_sentences(orig_text)
        sus_sents = make_sentences(sus_text)
        
        if len(orig_sents) == 0 or len(sus_sents) == 0:
            st.error("Could not parse enough sentences. Add longer text.")
        else:
            matches = []
            
            # loop through each sentence in suspect text
            for s_idx, s_sentence in enumerate(sus_sents):
                best_match = ""
                highest_sim = 0.0
                
                for o_sentence in orig_sents:
                    score = calc_score(s_sentence, o_sentence)
                    if score > highest_sim:
                        highest_sim = score
                        best_match = o_sentence
                
                # check against slider cutoff
                if highest_sim >= cutoff:
                    matches.append({
                        "suspect": s_sentence,
                        "original": best_match,
                        "score": highest_sim
                    })
            
            total_sus = len(sus_sents)
            flagged = len(matches)
            pct = (flagged / total_sus) * 100 if total_sus > 0 else 0
            
            st.divider()
            
            # KPI stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Plagiarism %", f"{pct:.1f}%")
            c2.metric("Flagged Sentences", f"{flagged} of {total_sus}")
            c3.metric("Threshold Cutoff", f"{cutoff}")
            
            st.subheader("Flagged Matches")
            if len(matches) == 0:
                st.info("No plagiarism detected above current threshold.")
            else:
                for idx, m in enumerate(matches, 1):
                    with st.expander(f"Match #{idx} (Similarity: {m['score']*100:.1f}%)"):
                        st.markdown(f"**Original:** {m['original']}")
                        st.markdown(f"**Suspect:** {m['suspect']}")
            
            st.divider()
            
            # Simple text report
            report_lines = [
                "PLAGIARISM AUDIT REPORT",
                f"Total Sentences Tested: {total_sus}",
                f"Flagged Sentences: {flagged}",
                f"Plagiarism Percentage: {pct:.2f}%",
                f"Cutoff Used: {cutoff}",
                "-" * 40,
                "MATCH DETAILS:"
            ]
            for m in matches:
                report_lines.append(f"[{m['score']*100:.1f}%] Suspect: {m['suspect']}")
                report_lines.append(f"        Original: {m['original']}\n")
            
            report_str = "\n".join(report_lines)
            
            st.download_button(
                label="Save Report as .txt",
                data=report_str,
                file_name="plagiarism_summary.txt",
                mime="text/plain"
            )
