import json
import networkx as nx
import spacy
import re

def main(dataset: dict = None) -> dict:
    """
    Builds a directed graph representing relationships between equations, 
    sentences, and mathematical symbols within papers. It extracts nodes 
    and edges based on contextual sentences surrounding each equation.
    
    Args:
        dataset (dict, optional): The dataset dictionary. If not provided, it will be loaded from file.
        
    Returns:
        dict: The updated dataset containing the processed graph relationship information.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Make sure it is installed.")
        return dataset

    we_loaded = False
    if dataset is None:
        try:
            with open("relationship_context.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            we_loaded = True
        except FileNotFoundError:
            print("relationship_context.json not found.")
            return None
    else:
        data = dataset

    # Create a directed graph
    G = nx.DiGraph()

    # Regex heuristic for identifying inline-math blocks in text
    math_pattern = re.compile(r'\S*(?:\\[a-zA-Z]+|[_^{}=<>≤≥±∇∂])\S*')

    from equation_context_builder import _governing_verb

    def _normalize_math(text: str) -> str:
        # Strip whitespace and curly braces for robust matching
        text = re.sub(r'[\s{}]+', '', text)
        # Strip trailing punctuation
        text = text.rstrip(',.;:')
        return text

    for paper_id, equations in data.items():
        if not isinstance(equations, dict):
            continue

        # To avoid duplicate sentence nodes across equations in the same paper
        seen_sentences = set()
        seen_maths = set()
        
        # Precompute normalized equations for this paper
        normalized_equations = {}
        for other_eq_id, other_eq_data in equations.items():
            if not isinstance(other_eq_data, dict):
                continue
            other_eq_text = other_eq_data.get("equation", "")
            if other_eq_text:
                normalized_equations[_normalize_math(other_eq_text)] = other_eq_id

        for eq_id, eq_data in equations.items():
            if not isinstance(eq_data, dict):
                continue
            eq_node_id = f"eq_{paper_id}_{eq_id}"
            
            # Extract attributes to store in the node
            equation_text = eq_data.get("equation", "")

            meaning_ctx = eq_data.get("meaning_context_window")
            meaning_ctx = meaning_ctx if isinstance(meaning_ctx, dict) else {}
            meaning_before = meaning_ctx.get("before_text", eq_data.get("before_text", ""))
            meaning_after = meaning_ctx.get("after_text", eq_data.get("after_text", ""))

            relation_ctx = eq_data.get("relation_graph_context_window")
            relation_ctx = relation_ctx if isinstance(relation_ctx, dict) else {}
            relation_before = relation_ctx.get("before_text", eq_data.get("before_text", ""))
            relation_after = relation_ctx.get("after_text", eq_data.get("after_text", ""))

            # 1. Add EQUATION node
            G.add_node(
                eq_node_id,
                node_type="equation",
                paper_id=paper_id,
                equation_id=eq_id,
                text=equation_text,
                meaning_before=meaning_before,
                meaning_after=meaning_after
            )

            # Extract sentences from the 2-sentence relation_graph_context_window
            full_relation_context = relation_before + " " + relation_after
            doc = nlp(full_relation_context)
            
            for i, sent in enumerate(doc.sents):
                sent_text = sent.text.strip()
                if not sent_text or sent_text in seen_sentences:
                    continue
                seen_sentences.add(sent_text)
                
                # 2. Add SENTENCE node
                sent_node_id = f"sent_{paper_id}_{len(seen_sentences)}"
                G.add_node(
                    sent_node_id,
                    node_type="sentence",
                    paper_id=paper_id,
                    text=sent_text
                )

                # Collect all entities in this sentence to link them together
                entities_in_sentence = []
                
                # Find main equations in the sentence
                for other_eq_id, other_eq_data in equations.items():
                    if not isinstance(other_eq_data, dict):
                        continue
                    other_eq_text = other_eq_data.get("equation", "")
                    if other_eq_text and other_eq_text in sent.text:
                        start_char = sent.text.find(other_eq_text)
                        
                        # Find token for predicate extraction
                        math_token = None
                        for token in sent:
                            rel_idx = token.idx - sent.start_char
                            if rel_idx >= start_char and rel_idx < start_char + len(other_eq_text):
                                math_token = token
                                break
                                
                        if not math_token:
                            math_token = sent[0]
                            
                        verb_info = _governing_verb(math_token, skip_formula_cues=True)
                        predicate = verb_info.get("verb_lemma", "")
                        
                        entities_in_sentence.append({
                            "node_id": f"eq_{paper_id}_{other_eq_id}",
                            "start_char": start_char,
                            "end_char": start_char + len(other_eq_text),
                            "predicate": predicate,
                            "is_inline": False
                        })

                # Extract inline math from this sentence
                raw_matches = list(math_pattern.finditer(sent.text))
                merged_maths = []
                
                if raw_matches:
                    curr_start = raw_matches[0].start()
                    curr_end = raw_matches[0].end()
                    
                    for i in range(1, len(raw_matches)):
                        next_match = raw_matches[i]
                        intervening_text = sent.text[curr_end:next_match.start()]
                        # Merge if separated by only whitespace or basic punctuation
                        if not intervening_text.strip() or intervening_text.strip() in [",", ".", ";", ":"]:
                            curr_end = next_match.end()
                        else:
                            merged_maths.append({
                                "start_char": curr_start, 
                                "end_char": curr_end, 
                                "text": sent.text[curr_start:curr_end]
                            })
                            curr_start = next_match.start()
                            curr_end = next_match.end()
                            
                    merged_maths.append({
                        "start_char": curr_start, 
                        "end_char": curr_end, 
                        "text": sent.text[curr_start:curr_end]
                    })
                
                for m in merged_maths:
                    start_char = m["start_char"]
                    end_char = m["end_char"]
                    
                    # Check if this overlaps with any found equations
                    overlaps = False
                    for ent in entities_in_sentence:
                        if start_char >= ent["start_char"] and end_char <= ent.get("end_char", -1):
                            overlaps = True
                            break
                    
                    if overlaps:
                        continue
                        
                    math_text = m["text"]
                    if not math_text:
                        continue
                        
                    normalized_math = _normalize_math(math_text)
                    if normalized_math in normalized_equations:
                        # Map to existing equation
                        matched_eq_id = normalized_equations[normalized_math]
                        math_node_id = f"eq_{paper_id}_{matched_eq_id}"
                    else:
                        if math_text in seen_maths:
                            continue
                        seen_maths.add(math_text)
                        
                        # 3. Add INLINE-MATH node
                        math_node_id = f"math_{paper_id}_{len(seen_maths)}"
                        G.add_node(
                            math_node_id,
                            node_type="inline-math",
                            paper_id=paper_id,
                            text=math_text
                        )

                    # Find token for predicate extraction
                    math_token = None
                    for token in sent:
                        rel_idx = token.idx - sent.start_char
                        if rel_idx >= start_char and rel_idx < end_char:
                            math_token = token
                            break
                    
                    if not math_token:
                        math_token = sent[0]
                        
                    verb_info = _governing_verb(math_token, skip_formula_cues=True)
                    predicate = verb_info.get("verb_lemma", "")
                    
                    entities_in_sentence.append({
                        "node_id": math_node_id,
                        "start_char": start_char,
                        "predicate": predicate,
                        "is_inline": True
                    })

                # Sort entities by their appearance in the sentence
                entities_in_sentence.sort(key=lambda x: x["start_char"])
                
                # Link sentence to Main Equation node
                G.add_edge(sent_node_id, eq_node_id, relation_type="context_of")
                
                # Find main equations in the sentence (Literal LaTeX string matches) -> Entity and "collapsed" from Main Equation -> Entity
                for ent in entities_in_sentence:
                    G.add_edge(sent_node_id, ent["node_id"], predicate="mentions")
                    G.add_edge(
                        eq_node_id, 
                        ent["node_id"], 
                        relation_type="dependency_reference",
                        grade="strong",
                        description="inline math match" if ent.get("is_inline") else "dependency reference",
                        evidence=sent_text,
                        dependency_evidence=ent["predicate"]
                    )

            # ----------------------------------------------------
            # Explicit Textual Equation References (Run on full_relation_context to bypass Spacy "Eqs." splits)
            # ----------------------------------------------------
            ref_pattern = r'(?i)\b(?:equations?|eqs?)\.?\s*((?:\d+|\s*,\s*|\s*and\s*|\s*&\s*|\(|\)|\s+)*)'
            for match in re.finditer(ref_pattern, full_relation_context):
                ref_str = match.group(1)
                ref_digits = re.findall(r'\d+', ref_str)
                
                start_idx = max(0, match.start() - 40)
                end_idx = min(len(full_relation_context), match.end() + 40)
                snippet = "..." + full_relation_context[start_idx:end_idx] + "..."
                
                for d_str in ref_digits:
                    if d_str in equations and d_str != eq_id:
                        ref_node = f"eq_{paper_id}_{d_str}"
                        # Edge points FROM referenced equation TO current equation
                        G.add_edge(ref_node, eq_node_id, 
                                   relation_type="textual_reference",
                                   grade="strong",
                                   description="explicit text reference",
                                   evidence=snippet)

        # ----------------------------------------------------
        # All-Pairs LHS/RHS Equation Analysis
        # ----------------------------------------------------
        def _safe_add_edge(u, v, **kwargs):
            if G.has_edge(u, v):
                existing = G.get_edge_data(u, v)
                # Never overwrite a textual_reference edge with a math relation, 
                # as textual references are explicit user-authored links.
                if existing.get("relation_type") == "textual_reference":
                    return
            G.add_edge(u, v, **kwargs)

        def _split_equation(eq_text):
            operators = ['=', '\\approx', '\\equiv', '<', '>', '\\leq', '\\geq', '\\sim', '\\in', '\\ni', '\\propto']
            
            level = 0
            for i in range(len(eq_text)):
                if eq_text[i] == '{':
                    level += 1
                elif eq_text[i] == '}':
                    level -= 1
                elif level == 0:
                    for op in operators:
                        if eq_text.startswith(op, i):
                            return eq_text[:i], eq_text[i+len(op):]
                            
            # Fallback to simple regex if nothing found at level 0
            match = re.search(r'(=|\\approx|\\equiv|<|>|\\leq|\\geq|\\sim|\\in)', eq_text)
            if match:
                idx = match.start()
                return eq_text[:idx], eq_text[match.end():]
                
            return "", eq_text

        def _extract_symbols(latex_text):
            modifiers_and_ops = [
                r'\\frac', r'\\sqrt', r'\\left', r'\\right', r'\\partial', r'\\int', r'\\sum', r'\\prod',
                r'\\vec', r'\\hat', r'\\bar', r'\\tilde', r'\\dot', r'\\ddot',
                r'\\mathbf', r'\\mathrm', r'\\mathit', r'\\mathcal', r'\\boldsymbol',
                r'\\pm', r'\\mp', r'\\cdot', r'\\times', r'\\div', r'\\nabla',
                r'\\sin', r'\\cos', r'\\tan', r'\\exp', r'\\log', r'\\ln',
                r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\\text\{.*?\}',
                r'\\llangle', r'\\rrangle', r'\\langle', r'\\rangle',
                r'\\bra', r'\\ket', r'\\braket', r'\\kern', r'\\vphantom', r'\\displaystyle',
                r'\\in', r'\\ni', r'\\approx', r'\\equiv', r'\\leq', r'\\geq', r'\\sim', r'\\propto'
            ]
            cleaned = latex_text
            for cmd in modifiers_and_ops:
                cleaned = re.sub(cmd, '', cleaned)
                
            symbols = set()
            for match in re.finditer(r'\\[a-zA-Z]+|[a-zA-Z]', cleaned):
                symbols.add(match.group())
            return {s for s in symbols if s.strip()}
        
        # Build symbol sets for all equations in this paper
        eq_symbols = {}
        for eq_id, eq_data in equations.items():
            if not isinstance(eq_data, dict) or "equation" not in eq_data:
                continue
            eq_text = eq_data.get("equation", "")
            lhs_text, rhs_text = _split_equation(eq_text)
            eq_symbols[eq_id] = {
                "eq_text": eq_text,
                "eq_norm": _normalize_math(eq_text),
                "lhs_raw": lhs_text,
                "rhs_raw": rhs_text,
                "lhs_norm": _normalize_math(lhs_text),
                "rhs_norm": _normalize_math(rhs_text),
                "lhs": _extract_symbols(lhs_text),
                "rhs": _extract_symbols(rhs_text)
            }
            
        # Compare all pairs of equations
        eq_ids = list(eq_symbols.keys())
        for i in range(len(eq_ids)):
            for j in range(len(eq_ids)):
                if i == j:
                    continue
                    
                eq_a_id = eq_ids[i]
                eq_b_id = eq_ids[j]
                
                sym_a = eq_symbols[eq_a_id]
                sym_b = eq_symbols[eq_b_id]
                
                node_a = f"eq_{paper_id}_{eq_a_id}"
                node_b = f"eq_{paper_id}_{eq_b_id}"
                
                # Rule 0: Exact Match & Same Expression
                if sym_a["eq_norm"] == sym_b["eq_norm"] and sym_a["eq_norm"]:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="strong", 
                               description="exact match")
                    continue
                    
                if sym_a["lhs_norm"] and sym_a["lhs_norm"] == sym_b["lhs_norm"] and sym_a["rhs_norm"] and sym_a["rhs_norm"] == sym_b["rhs_norm"]:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="strong", 
                               description="same expression, different operator")
                    continue
                
                # Rule 1: EQ_A RHS uses EQ_B's complete LHS
                a_rhs_b_lhs = False
                matched_b_lhs = []
                if len(sym_b["lhs"]) == 1:
                    b_sym = list(sym_b["lhs"])[0]
                    if b_sym in sym_a["rhs"]:
                        a_rhs_b_lhs = True
                        matched_b_lhs = [b_sym]
                else:
                    if sym_b["lhs_norm"] and sym_b["lhs_norm"] in sym_a["rhs_norm"]:
                        a_rhs_b_lhs = True
                        matched_b_lhs = [sym_b["lhs_raw"]]
                
                if a_rhs_b_lhs:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="strong", 
                               description="uses defined quantity", audit_trail=matched_b_lhs)

                # Rule 2: EQ_B RHS uses EQ_A's complete LHS (inverse check)
                a_lhs_b_rhs = False
                matched_a_lhs = []
                if len(sym_a["lhs"]) == 1:
                    a_sym = list(sym_a["lhs"])[0]
                    if a_sym in sym_b["rhs"]:
                        a_lhs_b_rhs = True
                        matched_a_lhs = [a_sym]
                else:
                    if sym_a["lhs_norm"] and sym_a["lhs_norm"] in sym_b["rhs_norm"]:
                        a_lhs_b_rhs = True
                        matched_a_lhs = [sym_a["lhs_raw"]]
                        
                if a_lhs_b_rhs:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="strong", 
                               description="defined quantity used by", audit_trail=matched_a_lhs)
                               
                # Rule 3: Both equations have the same LHS symbol
                a_lhs_b_lhs = False
                matched_same_lhs = []
                if sym_a["lhs_norm"] and sym_a["lhs_norm"] == sym_b["lhs_norm"]:
                    a_lhs_b_lhs = True
                    matched_same_lhs = [sym_a["lhs_raw"]]
                    
                if not a_rhs_b_lhs and not a_lhs_b_rhs and a_lhs_b_lhs:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="potential", 
                               description="shared LHS", audit_trail=matched_same_lhs)
                               
                # Rule 3.5: One LHS uses another LHS as a substring
                lhs_uses_lhs = False
                matched_lhs_uses = []
                description_lhs_uses = ""
                if not a_lhs_b_lhs:
                    if sym_b["lhs_norm"] and sym_b["lhs_norm"] in sym_a["lhs_norm"]:
                        lhs_uses_lhs = True
                        matched_lhs_uses = [sym_b["lhs_raw"]]
                        description_lhs_uses = "LHS uses defined quantity"
                    elif sym_a["lhs_norm"] and sym_a["lhs_norm"] in sym_b["lhs_norm"]:
                        lhs_uses_lhs = True
                        matched_lhs_uses = [sym_a["lhs_raw"]]
                        description_lhs_uses = "defined quantity used in LHS"

                if not a_rhs_b_lhs and not a_lhs_b_rhs and not a_lhs_b_lhs and lhs_uses_lhs:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="potential", 
                               description=description_lhs_uses, audit_trail=matched_lhs_uses)
                               
                # Rule 4: Same local context (fallback)
                if not a_rhs_b_lhs and not a_lhs_b_rhs and not a_lhs_b_lhs and not lhs_uses_lhs:
                    # Check if they share the same context block
                    ctx_a = eq_symbols[eq_a_id].get("full_context", "")
                    ctx_b = eq_symbols[eq_b_id].get("full_context", "")
                    if ctx_a and ctx_a == ctx_b:
                        _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="potential",
                                   description="same local context")
                               
                # Rule 5: They only share symbols (fallback)
                shared_all = sym_a["lhs"].union(sym_a["rhs"]).intersection(sym_b["lhs"].union(sym_b["rhs"]))
                if not a_rhs_b_lhs and not a_lhs_b_rhs and not a_lhs_b_lhs and not lhs_uses_lhs and shared_all:
                    _safe_add_edge(node_a, node_b, relation_type="lhs_rhs_relation", grade="potential", 
                               description="shared symbols", audit_trail=list(shared_all))

    # Save the graph to a GML file
    # output_file = "equation_graph.gml"
    # nx.write_gml(G, output_file)
    # print(f"Graph with {G.number_of_nodes()} nodes created and saved to {output_file}")
    
    # Export relations to JSON
    for paper_id, equations in data.items():
        for eq_id in equations.keys():
            node_id = f"eq_{paper_id}_{eq_id}"
            if not G.has_node(node_id):
                continue
                
            relations = []
            related_targets = set()
            for u, v, d in G.edges(node_id, data=True):
                target_node = v
                # We only want Equation -> Equation relations in this JSON
                if not target_node.startswith(f"eq_{paper_id}_"):
                    continue
                    
                target_eq_id = target_node.split("_")[-1]
                related_targets.add(target_eq_id)
                
                rel = {
                    "target_equation_id": target_eq_id,
                    "relation_type": d.get("relation_type", "dependency_reference" if "evidence" in d else "unknown"),
                    "grade": d.get("grade", "none"),
                    "description": d.get("description", "none")
                }
                if "audit_trail" in d:
                    rel["audit_trail"] = d["audit_trail"]
                if "evidence" in d:
                    rel["evidence"] = d["evidence"]
                    
                relations.append(rel)
                
            # Add explicit "none" relations for equations with no connection
            for other_eq_id in equations.keys():
                if not isinstance(equations[other_eq_id], dict) or "equation" not in equations[other_eq_id]:
                    continue
                if other_eq_id != eq_id and other_eq_id not in related_targets:
                    relations.append({
                        "target_equation_id": other_eq_id,
                        "grade": "none"
                    })
                
            # Sort relations numerically by target_equation_id
            def sort_key(x):
                # Try to extract the first integer block from the target_equation_id
                m = re.search(r'\d+', x["target_equation_id"])
                return int(m.group()) if m else float('inf')
                
            relations.sort(key=sort_key)
            equations[eq_id]["relations"] = relations
            
    # json_out_file = "relationship_context_with_relations.json"
    # with open(json_out_file, "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=2)
    # print(f"Relations JSON exported to {json_out_file}")

    return data

if __name__ == "__main__":
    main()
