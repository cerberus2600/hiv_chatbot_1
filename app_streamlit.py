# enhanced_hiv_chatbot.py
# Production-ready HIV Knowledge Graph Chatbot with Direct Neo4j Priority

import streamlit as st
import openai
from neo4j import GraphDatabase
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple, Set
import time
from datetime import datetime
import hashlib
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import re

# --- Configuration ---
NEO4J_URI = "neo4j+ssc://b921755a.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# Initialize connections
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Query Analyzer ---
class QueryAnalyzer:
    """Analyze and extract key terms from queries"""
    
    def __init__(self):
        # Comprehensive medical terms including variations
        self.medical_terms = {
            'treatment', 'therapy', 'medication', 'drug', 'medicine', 'antiretroviral', 
            'art', 'haart', 'prep', 'pep', 'truvada', 'descovy', 'tenofovir', 
            'emtricitabine', 'efavirenz', 'dolutegravir', 'raltegravir', 'protease',
            'inhibitor', 'integrase', 'reverse transcriptase', 'nnrti', 'nrti', 'pi',
            'first line', 'second line', 'first-line', 'second-line', 'regimen',
            'hiv', 'aids', 'cd4', 'viral load', 'transmission', 'prevention',
            'symptom', 'diagnosis', 'test', 'window period', 'seroconversion',
            'opportunistic', 'infection', 'resistance', 'adherence', 'side effect',
            'tdf', 'ftc', 'efv', 'dtg', 'ral', 'lpv', 'atv', 'drv'  # Drug abbreviations
        }
        
        # Expanded synonyms
        self.term_expansions = {
            'treatment': ['therapy', 'medication', 'drug', 'medicine', 'regimen', 'treats'],
            'first line': ['first-line', 'firstline', 'initial', 'preferred', 'recommended', 'first_line'],
            'second line': ['second-line', 'secondline', 'second_line', 'alternative'],
            'art': ['antiretroviral therapy', 'antiretroviral', 'haart', 'arvs', 'antiretrovirals'],
            'side effect': ['adverse effect', 'toxicity', 'reaction', 'adverse event'],
            'hiv': ['human immunodeficiency virus', 'hiv-1', 'hiv-2', 'hiv1', 'hiv2'],
            'aids': ['acquired immunodeficiency syndrome', 'advanced hiv']
        }
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract all possible search terms from query"""
        query_lower = query.lower()
        extracted_terms = []
        
        # Add original query words (excluding common stop words)
        stop_words = {'what', 'is', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'and', 'or'}
        query_words = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
        extracted_terms.extend(query_words)
        
        # Extract exact medical terms
        for term in self.medical_terms:
            if term in query_lower:
                extracted_terms.append(term)
                # Add expansions
                if term in self.term_expansions:
                    extracted_terms.extend(self.term_expansions[term])
        
        # Handle specific phrases with variations
        if 'first' in query_lower and ('line' in query_lower or 'treatment' in query_lower):
            extracted_terms.extend(['first line', 'first-line', 'firstline', 'first_line', 'initial treatment', 'preferred'])
        
        if 'second' in query_lower and ('line' in query_lower or 'treatment' in query_lower):
            extracted_terms.extend(['second line', 'second-line', 'secondline', 'second_line', 'alternative'])
        
        # Remove duplicates
        return list(dict.fromkeys(extracted_terms))

# --- Direct Neo4j Search Engine ---
class DirectNeo4jSearch:
    """Direct search in Neo4j without embeddings"""
    
    def __init__(self, driver):
        self.driver = driver
        self.query_analyzer = QueryAnalyzer()
    
    def comprehensive_search(self, query: str) -> Tuple[List[str], List[Dict], float]:
        """
        Comprehensive search that checks all possible fields and relationship types
        Returns: (entities, relationships, confidence)
        """
        key_terms = self.query_analyzer.extract_key_terms(query)
        
        entities = set()
        relationships = []
        
        # Debug: Show what terms we're searching for
        if st.session_state.get('debug_mode', False):
            st.sidebar.write("Search terms:", key_terms)
        
        try:
            with self.driver.session() as session:
                # Query 1: Broad entity search across all fields
                entity_search = """
                UNWIND $terms AS term
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower(term)
                   OR toLower(n.description) CONTAINS toLower(term)
                   OR toLower(n.category) CONTAINS toLower(term)
                   OR toLower(n.type) CONTAINS toLower(term)
                   OR ANY(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower(term))
                RETURN DISTINCT n.name AS entity, labels(n) AS labels
                LIMIT 30
                """
                
                result = session.run(entity_search, terms=key_terms)
                for record in result:
                    if record['entity']:
                        entities.add(record['entity'])
                
                # Query 2: Search in relationship types
                rel_type_search = """
                UNWIND $terms AS term
                MATCH (n)-[r]->(m)
                WHERE toLower(type(r)) CONTAINS toLower(term)
                   OR toLower(n.name) CONTAINS toLower(term)
                   OR toLower(m.name) CONTAINS toLower(term)
                RETURN DISTINCT n.name AS source, type(r) AS relation, m.name AS target
                LIMIT 30
                """
                
                result = session.run(rel_type_search, terms=key_terms)
                for record in result:
                    entities.add(record['source'])
                    entities.add(record['target'])
                    relationships.append({
                        'source': record['source'],
                        'relation': record['relation'],
                        'target': record['target'],
                        'strength': 'high'
                    })
                
                # Query 3: Special handling for treatment queries
                if any(term in key_terms for term in ['treatment', 'therapy', 'first', 'line', 'medication', 'drug']):
                    treatment_search = """
                    MATCH (n)-[r]->(m)
                    WHERE type(r) IN ['TREATS', 'TREATMENT', 'FIRST_LINE', 'FIRST_LINE_TREATMENT', 
                                      'SECOND_LINE', 'SECOND_LINE_TREATMENT', 'MEDICATION', 'THERAPY']
                       OR (toLower(type(r)) CONTAINS 'treat')
                       OR (toLower(type(r)) CONTAINS 'first' AND toLower(type(r)) CONTAINS 'line')
                       OR (toLower(n.name) CONTAINS 'treatment' OR toLower(m.name) CONTAINS 'treatment')
                       OR (toLower(n.name) CONTAINS 'first' AND toLower(n.name) CONTAINS 'line')
                    RETURN n.name AS source, type(r) AS relation, m.name AS target
                    LIMIT 20
                    """
                    
                    result = session.run(treatment_search)
                    for record in result:
                        entities.add(record['source'])
                        entities.add(record['target'])
                        relationships.append({
                            'source': record['source'],
                            'relation': record['relation'],
                            'target': record['target'],
                            'strength': 'high'
                        })
                
                # Query 4: If we found entities, get ALL their relationships
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
                    
                    result = session.run(all_relationships, entities=entity_list)
                    for record in result:
                        rel_exists = any(
                            r['source'] == record['source'] and 
                            r['target'] == record['target'] and 
                            r['relation'] == record['relation'] 
                            for r in relationships
                        )
                        if not rel_exists:
                            relationships.append({
                                'source': record['source'],
                                'relation': record['relation'],
                                'target': record['target'],
                                'strength': record.get('strength', 'medium'),
                                'evidence': record.get('evidence', 'established')
                            })
                
                # Query 5: Fuzzy match for entities if we still don't have results
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
                    
                    result = session.run(fuzzy_search, terms=key_terms[:5])
                    for record in result:
                        if record['entity']:
                            entities.add(record['entity'])
                
        except Exception as e:
            st.error(f"Neo4j search error: {e}")
        
        # Calculate confidence based on results quality
        confidence = self._calculate_confidence(list(entities), relationships, key_terms)
        
        return list(entities), relationships, confidence
    
    def _calculate_confidence(self, entities: List[str], relationships: List[Dict], 
                            key_terms: List[str]) -> float:
        """
        Calculate confidence score:
        - 0.0-0.3: Low (few or no results)
        - 0.3-0.7: Medium (some relevant results)
        - 0.7-1.0: High (many relevant results with good term coverage)
        """
        if not entities and not relationships:
            return 0.0
        
        # Factor 1: Number of entities found (max 0.3)
        entity_score = min(len(entities) / 10, 1.0) * 0.3
        
        # Factor 2: Number of relationships found (max 0.3)
        relationship_score = min(len(relationships) / 10, 1.0) * 0.3
        
        # Factor 3: Term coverage - how many search terms appear in results (max 0.4)
        all_text = ' '.join(entities).lower()
        for rel in relationships:
            all_text += f" {rel['source']} {rel['relation']} {rel['target']}".lower()
        
        covered_terms = sum(1 for term in key_terms if term in all_text)
        term_coverage = (covered_terms / max(len(key_terms), 1)) * 0.4
        
        total_confidence = entity_score + relationship_score + term_coverage
        
        return min(total_confidence, 1.0)

# --- Response Generator ---
class ResponseGenerator:
    def __init__(self, client): self.client = client

    def generate_response(self, query, entities, relationships, confidence):
        relationship_facts = []
        for rel in relationships[:20]:
            fact = f"- {rel['source']} {rel['relation']} {rel['target']}"
            if rel.get('strength') == 'high': fact += " [VERIFIED]"
            relationship_facts.append(fact)

        if relationships or entities:
            system_msg = ("You are an expert HIV/AIDS medical assistant. "
                          "Use the provided knowledge as your PRIMARY source; "
                          "supplement only if incomplete. Never reveal the source.")
            context = (
                f"Entities: {', '.join(entities[:15]) or 'None'}\n"
                f"Relationships:\n" + ("\n".join(relationship_facts) or "None")
            )
        else:
            system_msg = ("You are an expert HIV/AIDS medical assistant. "
                          "Provide accurate, comprehensive information.")
            context = "Answer using your medical knowledge."

        full_prompt = f"Question: {query}\n\n{context}"

        try:
            resp = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": full_prompt},
                ],
                max_completion_tokens=600,   # correct param for Chat Completions
            )
            text = resp.choices[0].message.content
            if not text:
                raise RuntimeError("Empty text from model")
            return text
        except Exception as e:
            # fallback
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": full_prompt},
                    ],
                    max_completion_tokens=500,
                )
                return resp.choices[0].message.content or f"Error: {e}"
            except Exception as e2:
                return f"Error generating response: {e2}"

# --- Visualization ---
def create_relationship_graph(relationships: List[Dict], entities: List[str]):
    """Create interactive relationship visualization"""
    if not relationships:
        return None
    
    # Build graph data
    nodes = set()
    edges = []
    
    for rel in relationships[:15]:
        nodes.add(rel['source'])
        nodes.add(rel['target'])
        edges.append((rel['source'], rel['target'], rel['relation']))
    
    # Add standalone entities
    for entity in entities[:10]:
        nodes.add(entity)
    
    if len(nodes) < 2:
        return None
    
    import networkx as nx
    G = nx.DiGraph()
    
    for source, target, relation in edges:
        G.add_edge(source, target, relation=relation)
    
    for node in nodes:
        if node not in G:
            G.add_node(node)
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1.5, color='#888'),
            hoverinfo='none'
        ))
    
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[node[:30] for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=10),
        hoverinfo='text',
        marker=dict(
            size=15,
            color='#FF6B6B',
            line=dict(color='darkred', width=2)
        )
    )
    
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            title="Knowledge Graph Connections"
        )
    )
    
    return fig

# --- Get KG Statistics ---
@st.cache_data(ttl=300)
def get_kg_statistics():
    """Get knowledge graph statistics"""
    stats = {}
    
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            stats['entities'] = result.single()['count']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            stats['relationships'] = result.single()['count']
            
    except Exception as e:
        stats = {'entities': 'N/A', 'relationships': 'N/A'}
    
    return stats

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="HIV HelpMate",
        page_icon="🧬",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .relationship-box {
        background-color: #f7f7f7;
        border-left: 3px solid #FF6B6B;
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 14px;
    }
    .entity-tag {
        background-color: #FFE5E5;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        display: inline-block;
        font-size: 13px;
    }
    .confidence-indicator {
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        font-size: 12px;
        font-weight: bold;
    }
    .confidence-high { background-color: #d4edda; color: #155724; }
    .confidence-medium { background-color: #fff3cd; color: #856404; }
    .confidence-low { background-color: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧬 HIV HelpMate")
    
    # Initialize systems
    neo4j_search = DirectNeo4jSearch(driver)
    generator = ResponseGenerator(client)
    
    # Get statistics
    kg_stats = get_kg_statistics()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Knowledge Graph")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entities", kg_stats.get('entities', 'N/A'))
        with col2:
            st.metric("Relationships", kg_stats.get('relationships', 'N/A'))
        
        st.markdown("---")
        
        # Settings
        show_graph = st.checkbox("Show Graph Visualization", value=True)
        show_raw_data = st.checkbox("Show Raw KG Data", value=False)
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        
        # Confidence explanation
        with st.expander("📊 About Confidence Score"):
            st.markdown("""
            **Confidence indicates how well the query matches the knowledge graph:**
            
            🟢 **High (70-100%)**: Many relevant entities and relationships found
            
            🟡 **Medium (30-70%)**: Some relevant data found
            
            🔴 **Low (0-30%)**: Limited or no specific data found
            
            The score is based on:
            - Number of matching entities
            - Number of relevant relationships
            - Coverage of search terms in results
            """)
        
        if st.button("🔄 Clear Chat"):
            st.session_state.clear()
            st.rerun()
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Welcome! I'm your HIV Knowledge Assistant. Ask me any question about HIV/AIDS."
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about HIV/AIDS..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant"):
            # Search knowledge graph
            with st.spinner("🔍 Searching knowledge graph..."):
                start_time = time.time()
                entities, relationships, confidence = neo4j_search.comprehensive_search(prompt)
                retrieval_time = time.time() - start_time
            
            # Display retrieved knowledge graph data
            st.markdown("### 📚 Knowledge Graph Results")
            
            # Show confidence level with explanation
            conf_class = "high" if confidence > 0.7 else "medium" if confidence > 0.3 else "low"
            conf_percent = f"{confidence:.0%}"
            st.markdown(
                f'<div class="confidence-indicator confidence-{conf_class}">'
                f'Confidence: {conf_percent}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Create two columns for entities and relationships
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**🎯 Entities Found:**")
                if entities:
                    for entity in entities[:12]:
                        st.markdown(f'<div class="entity-tag">{entity}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No specific entities found")
            
            with col2:
                st.markdown("**🔗 Relationships:**")
                if relationships:
                    # Group relationships by type
                    rel_groups = {}
                    for rel in relationships:
                        rel_type = rel['relation']
                        if rel_type not in rel_groups:
                            rel_groups[rel_type] = []
                        rel_groups[rel_type].append(rel)
                    
                    # Display grouped relationships
                    for rel_type, rels in list(rel_groups.items())[:6]:
                        with st.expander(f"{rel_type} ({len(rels)} found)", expanded=len(rel_groups) <= 3):
                            for rel in rels[:5]:
                                strength_emoji = "🔴" if rel.get('strength') == 'high' else "🟡"
                                st.markdown(
                                    f'<div class="relationship-box">'
                                    f'{strength_emoji} {rel["source"]} → {rel["target"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                else:
                    st.caption("No specific relationships found")
            
            # Show graph visualization if enabled
            if show_graph and (relationships or entities):
                fig = create_relationship_graph(relationships, entities)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data if enabled
            if show_raw_data:
                with st.expander("🔍 Raw Knowledge Graph Data"):
                    if entities:
                        st.markdown("**Entities:**")
                        st.json(entities[:10])
                    if relationships:
                        st.markdown("**Relationships:**")
                        st.json([{
                            'source': r['source'],
                            'relation': r['relation'],
                            'target': r['target'],
                            'strength': r.get('strength', 'N/A')
                        } for r in relationships[:10]])
            
            # Generate response
            st.markdown("### 💬 Response")
            with st.spinner("Generating response..."):
                response = generator.generate_response(prompt, entities, relationships, confidence)
            
            st.markdown(response)
            
            # Show metrics
            st.caption(f"⏱️ Search time: {retrieval_time:.2f}s | 📊 Found: {len(entities)} entities, {len(relationships)} relationships")
            
            # Save to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    main()
