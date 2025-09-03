
import streamlit as st
import openai
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import pandas as pd
import plotly.graph_objects as go
import re

# --- Configuration ---
NEO4J_URI = "neo4j+ssc://b921755a.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"] 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize connections
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ---------- Helpers ----------
# Only display labels that start with an alphabetic character (filters "1.", "(...)")
NODE_DISPLAY_REGEX = re.compile(r"^[A-Za-z]")

def is_displayable_node(name: Optional[str]) -> bool:
    if not name or not isinstance(name, str):
        return False
    return bool(NODE_DISPLAY_REGEX.match(name.strip()))

def filter_entities_for_display(entities: List[str]) -> List[str]:
    return [e for e in entities if is_displayable_node(e)]

def filter_relationships_for_display(rels: List[Dict]) -> List[Dict]:
    out = []
    for r in rels:
        if is_displayable_node(r.get("source")) and is_displayable_node(r.get("target")):
            out.append(r)
    return out

# --- Query Analyzer ---
class QueryAnalyzer:
    """Analyze queries and decide if they are HIV-related; extract key terms."""
    def __init__(self):
        # HIV-centric vocabulary (names, acronyms, therapies, labs)
        self.hiv_terms = {
            'hiv', 'aids', 'antiretroviral', 'arv', 'art', 'haart', 'prep', 'pep',
            'cd4', 'viral load', 'u=u', 'undetectable', 'seroconversion', 'window period',
            'opportunistic infection', 'oi', 'p24', 'gp120', 'integrase', 'protease',
            'reverse transcriptase', 'nnrti', 'nrti', 'pi',
            # common drug names & combos used in your KG
            'tenofovir', 'emtricitabine', 'truvada', 'descovy', 'efavirenz',
            'dolutegravir', 'raltegravir', 'lopinavir', 'ritonavir', 'darunavir',
        }

        # Broader medical terms used to expand searches
        self.medical_terms = {
            'treatment', 'therapy', 'medication', 'drug', 'regimen', 'prevention',
            'adherence', 'resistance', 'diagnosis', 'symptom', 'test'
        }

    def is_hiv_related(self, query: str) -> Tuple[bool, float, List[str]]:
        """
        Returns (is_hiv, score, matched_terms)
        Simple heuristic:
          - If 'hiv' or 'aids' present -> True (score high)
          - Else: count matches with hiv_terms & medical_terms; threshold on count
        """
        q = query.lower()
        matched = set()

        # direct hit
        if 'hiv' in q or 'aids' in q:
            # also collect specific term matches for transparency
            for t in self.hiv_terms:
                if t in q:
                    matched.add(t)
            return True, 1.0, sorted(matched) if matched else ['hiv/aids']

        # otherwise accumulate matches
        for t in self.hiv_terms.union(self.medical_terms):
            if t in q:
                matched.add(t)

        # Heuristic: if ≥2 HIV-specific terms or ≥1 HIV-specific + ≥1 medical term
        hiv_hits = sum(1 for t in matched if t in self.hiv_terms)
        med_hits = sum(1 for t in matched if t in self.medical_terms)

        is_hiv = hiv_hits >= 2 or (hiv_hits >= 1 and med_hits >= 1)
        score = min((hiv_hits * 0.6) + (med_hits * 0.2), 1.0) if is_hiv else 0.0

        return is_hiv, score, sorted(matched)

    def extract_key_terms(self, query: str) -> List[str]:
        q = query.lower()
        extracted = []

        stop_words = {'what','is','the','a','an','of','for','in','on','at','to','and','or'}
        words = [w for w in re.split(r"\W+", q) if w and w not in stop_words and len(w) > 2]
        extracted.extend(words)

        # add phrases
        for phrase in list(self.hiv_terms) + list(self.medical_terms):
            if phrase in q:
                extracted.append(phrase)

        # special forms
        if 'first' in q and ('line' in q or 'treatment' in q):
            extracted.extend(['first line', 'first-line', 'initial treatment', 'preferred'])
        if 'second' in q and ('line' in q or 'treatment' in q):
            extracted.extend(['second line', 'second-line', 'alternative'])

        # dedupe, keep order
        seen = {}
        for t in extracted:
            seen.setdefault(t, None)
        return list(seen.keys())

# --- Direct Neo4j Search Engine ---
class DirectNeo4jSearch:
    """Direct search in Neo4j without embeddings"""
    def __init__(self, driver):
        self.driver = driver
        self.analyzer = QueryAnalyzer()

    def comprehensive_search(self, query: str) -> Tuple[List[str], List[Dict], float]:
        key_terms = self.analyzer.extract_key_terms(query)
        entities = set()
        relationships = []

        if st.session_state.get('debug_mode', False):
            st.sidebar.write("Search terms:", key_terms)

        try:
            with self.driver.session() as session:
                # 1) Broad entity search
                entity_search = """
                UNWIND $terms AS term
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower(term)
                   OR toLower(n.description) CONTAINS toLower(term)
                   OR toLower(n.category) CONTAINS toLower(term)
                   OR toLower(n.type) CONTAINS toLower(term)
                   OR ANY(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower(term))
                RETURN DISTINCT n.name AS entity
                LIMIT 30
                """
                for record in session.run(entity_search, terms=key_terms):
                    if record['entity']:
                        entities.add(record['entity'])

                # 2) Relationship search
                rel_type_search = """
                UNWIND $terms AS term
                MATCH (n)-[r]->(m)
                WHERE toLower(type(r)) CONTAINS toLower(term)
                   OR toLower(n.name) CONTAINS toLower(term)
                   OR toLower(m.name) CONTAINS toLower(term)
                RETURN DISTINCT n.name AS source, type(r) AS relation, m.name AS target
                LIMIT 30
                """
                for rec in session.run(rel_type_search, terms=key_terms):
                    entities.add(rec['source']); entities.add(rec['target'])
                    relationships.append({
                        'source': rec['source'],
                        'relation': rec['relation'],
                        'target': rec['target'],
                        'strength': 'high'
                    })

                # 3) Treatment-focused relations
                if any(t in key_terms for t in ['treatment','therapy','first','line','medication','drug']):
                    treatment_search = """
                    MATCH (n)-[r]->(m)
                    WHERE type(r) IN ['TREATS','TREATMENT','FIRST_LINE','FIRST_LINE_TREATMENT',
                                      'SECOND_LINE','SECOND_LINE_TREATMENT','MEDICATION','THERAPY']
                       OR toLower(type(r)) CONTAINS 'treat'
                       OR (toLower(type(r)) CONTAINS 'first' AND toLower(type(r)) CONTAINS 'line')
                       OR toLower(n.name) CONTAINS 'treatment' OR toLower(m.name) CONTAINS 'treatment'
                    RETURN n.name AS source, type(r) AS relation, m.name AS target
                    LIMIT 20
                    """
                    for rec in session.run(treatment_search):
                        entities.add(rec['source']); entities.add(rec['target'])
                        relationships.append({
                            'source': rec['source'],
                            'relation': rec['relation'],
                            'target': rec['target'],
                            'strength': 'high'
                        })

                # 4) Expand all relations for found entities
                if entities:
                    entity_list = list(entities)[:20]
                    all_relationships = """
                    MATCH (n)-[r]->(m)
                    WHERE n.name IN $entities OR m.name IN $entities
                    RETURN n.name AS source, type(r) AS relation, m.name AS target,
                           r.strength AS strength, r.evidence AS evidence
                    ORDER BY 
                        CASE 
                            WHEN type(r) CONTAINS 'FIRST' THEN 1
                            WHEN type(r) CONTAINS 'TREAT' THEN 2
                            WHEN type(r) = 'CAUSES' THEN 3
                            WHEN type(r) = 'PREVENTS' THEN 4
                            ELSE 5
                        END
                    LIMIT 50
                    """
                    for rec in session.run(all_relationships, entities=entity_list):
                        exists = any(
                            r['source']==rec['source'] and r['target']==rec['target'] and r['relation']==rec['relation']
                            for r in relationships
                        )
                        if not exists:
                            relationships.append({
                                'source': rec['source'],
                                'relation': rec['relation'],
                                'target': rec['target'],
                                'strength': rec.get('strength', 'medium') if hasattr(rec, "get") else 'medium',
                                'evidence': rec.get('evidence', 'established') if hasattr(rec, "get") else 'established'
                            })

                # 5) Fuzzy expansion if sparse
                if len(entities) < 5:
                    fuzzy_search = """
                    MATCH (n)
                    WHERE ANY(term IN $terms WHERE 
                        toLower(n.name) =~ ('.*' + toLower(term) + '.*')
                        OR ANY(label IN labels(n) WHERE toLower(label) CONTAINS toLower(term))
                    )
                    RETURN DISTINCT n.name AS entity
                    LIMIT 15
                    """
                    for rec in session.run(fuzzy_search, terms=key_terms[:5]):
                        if rec['entity']:
                            entities.add(rec['entity'])

        except Exception as e:
            st.error(f"Neo4j search error: {e}")

        confidence = self._calculate_confidence(list(entities), relationships, key_terms)
        return list(entities), relationships, confidence

    def _calculate_confidence(self, entities: List[str], relationships: List[Dict], key_terms: List[str]) -> float:
        if not entities and not relationships:
            return 0.0
        entity_score = min(len(entities)/10, 1.0) * 0.3
        rel_score = min(len(relationships)/10, 1.0) * 0.3
        all_text = ' '.join(entities).lower() + ' ' + ' '.join(
            f"{r['source']} {r['relation']} {r['target']}".lower() for r in relationships
        )
        covered = sum(1 for t in key_terms if t in all_text)
        term_score = (covered / max(len(key_terms), 1)) * 0.4
        return min(entity_score + rel_score + term_score, 1.0)

# --- Response Generator ---
class ResponseGenerator:
    def __init__(self, client): self.client = client

    def generate_response(self, query, entities, relationships, confidence, model_name: str):
        # Filter for DEMO display & context cleanliness
        disp_entities = filter_entities_for_display(entities)[:15]
        disp_relationships = filter_relationships_for_display(relationships)[:20]

        rel_lines = []
        for rel in disp_relationships:
            line = f"- {rel['source']} {rel['relation']} {rel['target']}"
            if rel.get('strength') == 'high':
                line += " [VERIFIED]"
            rel_lines.append(line)

        if disp_relationships or disp_entities:
            system_msg = (
                "You are an expert HIV/AIDS medical assistant. "
                "Use the provided knowledge as your PRIMARY source; "
                "supplement only if incomplete. Never reveal the source."
            )
            context = f"Entities: {', '.join(disp_entities) or 'None'}\nRelationships:\n" + ("\n".join(rel_lines) or "None")
        else:
            system_msg = (
                "You are an expert HIV/AIDS medical assistant. "
                "Provide accurate, comprehensive information."
            )
            context = "Answer using your medical knowledge."

        full_prompt = f"Question: {query}\n\n{context}"

        try:
            resp = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=600,
            )
            txt = resp.choices[0].message.content
            if not txt:
                raise RuntimeError("Empty text from model")
            return txt
        except Exception as e:
            # fallback to the other model
            fallback = "gpt-4.1" if model_name == "gpt-4.1-mini" else "gpt-4.1-mini"
            try:
                resp = self.client.chat.completions.create(
                    model=fallback,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": full_prompt},
                    ],
                    max_tokens=500,
                )
                return resp.choices[0].message.content or f"Error: {e}"
            except Exception as e2:
                return f"Error generating response: {e2}"

# --- Visualization (filtered) ---
def create_relationship_graph(relationships: List[Dict], entities: List[str]):
    if not relationships and not entities:
        return None

    disp_rels = filter_relationships_for_display(relationships)[:15]
    disp_entities = filter_entities_for_display(entities)[:10]

    nodes = set()
    edges = []
    for r in disp_rels:
        nodes.add(r['source']); nodes.add(r['target'])
        edges.append((r['source'], r['target'], r['relation']))
    for e in disp_entities:
        nodes.add(e)

    if len(nodes) < 2 and not edges:
        return None

    import networkx as nx
    G = nx.DiGraph()
    for s, t, rel in edges: G.add_edge(s, t, relation=rel)
    for n in nodes:
        if n not in G: G.add_node(n)

    pos = nx.spring_layout(G, k=2, iterations=50)
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(x=[x0,x1,None], y=[y0,y1,None], mode='lines',
                                      line=dict(width=1.5, color='#888'), hoverinfo='none'))

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        text=[n[:30] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10),
        hoverinfo='text',
        marker=dict(size=15, color='#FF6B6B', line=dict(color='darkred', width=2))
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400, title="Knowledge Graph Connections"
        )
    )
    return fig

# --- KG Stats ---
@st.cache_data(ttl=300)
def get_kg_statistics():
    stats = {}
    try:
        with driver.session() as session:
            stats['entities'] = session.run("MATCH (n) RETURN count(n) AS c").single()['c']
            stats['relationships'] = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()['c']
    except Exception:
        stats = {'entities': 'N/A', 'relationships': 'N/A'}
    return stats

# --- Main App ---
def main():
    st.set_page_config(page_title="HIV HelpMate", page_icon="🧬", layout="wide")

    # CSS
    st.markdown("""
    <style>
    .relationship-box { background-color:#f7f7f7; border-left:3px solid #FF6B6B;
        padding:8px; margin:4px 0; border-radius:4px; font-size:14px; }
    .entity-tag { background-color:#FFE5E5; padding:4px 8px; margin:2px; border-radius:4px;
        display:inline-block; font-size:13px; }
    .confidence-indicator { padding:4px 8px; border-radius:4px; display:inline-block;
        font-size:12px; font-weight:bold; }
    .confidence-high { background-color:#d4edda; color:#155724; }
    .confidence-medium { background-color:#fff3cd; color:#856404; }
    .confidence-low { background-color:#f8d7da; color:#721c24; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧬 HIV HelpMate")

    neo4j_search = DirectNeo4jSearch(driver)
    generator = ResponseGenerator(client)
    analyzer = QueryAnalyzer()

    # Sidebar
    kg_stats = get_kg_statistics()
    with st.sidebar:
        st.markdown("### 📊 Knowledge Graph")
        c1, c2 = st.columns(2)
        with c1: st.metric("Entities", kg_stats.get('entities', 'N/A'))
        with c2: st.metric("Relationships", kg_stats.get('relationships', 'N/A'))
        st.markdown("---")

        # Model toggle
        use_gpt41 = st.checkbox("Enable GPT-4.1 (Quality Mode)", value=False,
                                help="ON = GPT-4.1, OFF = GPT-4.1 mini")
        selected_model = "gpt-4.1" if use_gpt41 else "gpt-4.1-mini"
        st.caption(f"Current model: **{selected_model}**")

        show_graph = st.checkbox("Show Graph Visualization", value=True)
        show_raw = st.checkbox("Show Raw KG Data", value=False)
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)

        with st.expander("📊 About Confidence Score"):
            st.markdown("""
            **Confidence shows how well the query matched the KG:**
            🟢 High (70–100%) · 🟡 Medium (30–70%) · 🔴 Low (0–30%)
            Factors: #entities, #relationships, and search-term coverage.
            """)

        if st.button("🔄 Clear Chat"):
            st.session_state.clear(); st.rerun()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role":"assistant",
            "content":"👋 Welcome! I'm your HIV Knowledge Assistant. Ask me any question about HIV/AIDS."
        }]

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    if prompt := st.chat_input("Ask about HIV/AIDS..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # Decide whether to use KG
            is_hiv, hiv_score, matched = analyzer.is_hiv_related(prompt)

            if is_hiv:
                with st.spinner("🔍 Searching knowledge graph..."):
                    t0 = time.time()
                    entities, relationships, confidence = neo4j_search.comprehensive_search(prompt)
                    retrieval_time = time.time() - t0

                st.markdown("### 📚 Knowledge Graph Results")
                conf_class = "high" if confidence > 0.7 else "medium" if confidence > 0.3 else "low"
                st.markdown(
                    f'<div class="confidence-indicator confidence-{conf_class}">Confidence: {confidence:.0%}</div>',
                    unsafe_allow_html=True
                )

                disp_entities = filter_entities_for_display(entities)
                disp_relationships = filter_relationships_for_display(relationships)

                col1, col2 = st.columns([1,2])
                with col1:
                    st.markdown("**🎯 Entities Found:**")
                    if disp_entities:
                        for e in disp_entities[:12]:
                            st.markdown(f'<div class="entity-tag">{e}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("No specific entities found")

                with col2:
                    st.markdown("**🔗 Relationships:**")
                    if disp_relationships:
                        groups: Dict[str,List[Dict]] = {}
                        for r in disp_relationships:
                            groups.setdefault(r['relation'], []).append(r)
                        for rel_type, rels in list(groups.items())[:6]:
                            with st.expander(f"{rel_type} ({len(rels)} found)", expanded=len(groups) <= 3):
                                for r in rels[:5]:
                                    emo = "🔴" if r.get('strength') == 'high' else "🟡"
                                    st.markdown(f'<div class="relationship-box">{emo} {r["source"]} → {r["target"]}</div>',
                                                unsafe_allow_html=True)
                    else:
                        st.caption("No specific relationships found")

                if show_graph and (disp_relationships or disp_entities):
                    fig = create_relationship_graph(disp_relationships, disp_entities)
                    if fig: st.plotly_chart(fig, use_container_width=True)

                if show_raw:
                    with st.expander("🔍 Raw Knowledge Graph Data (Filtered)"):
                        if disp_entities:
                            st.markdown("**Entities:**"); st.json(disp_entities[:10])
                        if disp_relationships:
                            st.markdown("**Relationships:**")
                            st.json([{'source':r['source'],'relation':r['relation'],'target':r['target'],
                                      'strength':r.get('strength','N/A')} for r in disp_relationships[:10]])

                # Generate response with selected model and KG context
                st.markdown("### 💬 Response")
                with st.spinner(f"Generating response with **{selected_model}**..."):
                    answer = generator.generate_response(prompt, entities, relationships, confidence, selected_model)
                st.markdown(answer)

                st.caption(
                    f"⏱️ Search time: {retrieval_time:.2f}s | "
                    f"📊 Found: {len(disp_entities)} entities, {len(disp_relationships)} relationships | "
                    f"🤖 Model: {selected_model}"
                )
            else:
                # Non-HIV: Skip KG entirely
                st.markdown("### 📚 Knowledge Graph Results")
                st.caption("KG search skipped (query not detected as HIV-related).")
                entities, relationships, confidence = [], [], 0.0

                st.markdown("### 💬 Response")
                with st.spinner(f"Generating response with **{selected_model}**..."):
                    answer = generator.generate_response(prompt, entities, relationships, confidence, selected_model)
                st.markdown(answer)

                st.caption(f"🤖 Model: {selected_model} | 🔎 KG: skipped | 🔗 Matched terms: {', '.join(matched) if matched else 'none'}")

            # persist assistant message
            st.session_state.messages.append({"role":"assistant","content":answer})

if __name__ == "__main__":
    main()
