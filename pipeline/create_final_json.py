import json
import re
import os

def main(dataset: dict = None, sentence_splits: dict = None, source_data: dict = None, predefined: dict = None) -> dict:
    """
    Consolidates processed data from various pipeline stages into a unified final JSON structure.
    It combines equations, their extracted meaning, symbols and their definitions (with source audit trails), 
    and equation relationships into the final `final.json` format.
    
    Args:
        dataset (dict, optional): The main processed dataset dictionary.
        sentence_splits (dict, optional): Dictionary containing equation meanings.
        source_data (dict, optional): Dictionary containing symbol definition sources.
        predefined (dict, optional): Predefined backup definitions.
        
    Returns:
        dict: The finalized output dataset dictionary.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    we_loaded = False
    
    if dataset is None:
        try:
            with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
                dataset = json.load(f)
        except Exception:
            pass
        we_loaded = True
            
    if sentence_splits is None:
        try:
            with open(os.path.join(base_dir, 'equation_sentence_splits.json'), 'r') as f:
                sentence_splits = json.load(f)
        except Exception:
            sentence_splits = {}
            
    # relationship_context is dataset, since graph_relationship modified it directly in memory
    relationship_context = dataset
        
    if predefined is None:
        try:
            with open(os.path.join(base_dir, 'predefined.json'), 'r') as f:
                predefined = json.load(f)
        except Exception:
            predefined = {}
        
    if source_data is None:
        try:
            with open(os.path.join(base_dir, 'definition_sources.json'), 'r') as f:
                source_data = json.load(f)
        except Exception:
            source_data = {}

    predefined_values = set(predefined.values())
        
    final_data = {}
    
    for paper_id, eq_dict in dataset.items():
        if not isinstance(eq_dict, dict):
            continue
            
        final_data[paper_id] = {}
        
        for eq_id, data in eq_dict.items():
            if not isinstance(data, dict) or "equation" not in data:
                continue
                
            equation = data.get("equation", "")
            
            # Meaning
            meaning = ""
            if paper_id in sentence_splits and eq_id in sentence_splits[paper_id]:
                meaning = sentence_splits[paper_id][eq_id].get("meaning", "")
                
            # Symbols
            symbols = data.get("symbols", {})
            if isinstance(symbols, list):
                symbols = {sym: "" for sym in symbols}
            
            # Relations
            relations_out = {}
            relations_in = []
            if paper_id in relationship_context and eq_id in relationship_context[paper_id]:
                relations_in = relationship_context[paper_id][eq_id].get("relations", [])
            
            for rel in relations_in:
                if rel.get("grade"):
                    target_id = rel.get("target_equation_id")
                    grade = rel.get("grade", "")
                    desc = rel.get("description", "")
                    if grade == "none" and not desc:
                        desc = "none"
                    relations_out[target_id] = {
                        "grade": grade,
                        "description": desc
                    }
            
            # Audit trail
            audit_trail = {
                "method extract_eq": f"Found equation {equation}"
            }
            
            extract_relation_list = []
            for rel in relations_in:
                if rel.get("grade"):
                    target_id = str(rel.get("target_equation_id")).strip('\'"')
                    grade = str(rel.get("grade", "")).strip('\'"')
                    desc = str(rel.get("description", "")).strip('\'"')
                    if grade == "none" and not desc:
                        desc = "none"
                    extract_relation_list.append(f"target_equation_id: {target_id} --> {grade} --> {desc}")
            
            find_symbols_list = []
            extract_symbol_list = []
            
            # Source data for symbols
            sym_sources = {}
            if source_data and paper_id in source_data and eq_id in source_data[paper_id]:
                sym_sources = source_data[paper_id][eq_id].get("symbol_definition_sources", {})

            for sym, desc in symbols.items():
                if not desc:
                    # Skip audit entries entirely if there is no extracted text for the symbol
                    continue
                    
                sym_clean = str(sym).strip('\'"')
                desc_clean = str(desc).strip('\'"')
                
                find_symbols_list.append(f"found {sym_clean}")
                source_info = sym_sources.get(sym, {})
                
                is_predefined = (source_info.get("source") == "backup_dictionary" and "predefined.json" in source_info.get("predefined_path", ""))
                
                if is_predefined:
                    extract_symbol_list.append(f"found {sym_clean} in predefined.json file --> {sym_clean}: {desc_clean}")
                else:
                    extract_symbol_list.append(f"extracted text {desc_clean} --> {sym_clean}: {desc_clean}")
                    
            for i, val in enumerate(find_symbols_list):
                audit_trail[f"__find_symbol_{i}__"] = val
            for i, val in enumerate(extract_symbol_list):
                audit_trail[f"__extract_symbol_name_{i}__"] = val
            for i, val in enumerate(extract_relation_list):
                audit_trail[f"__extract_relation_{i}__"] = val
            
            final_data[paper_id][eq_id] = {
                "equation": equation,
                "meaning": meaning,
                "symbols": symbols,
                "relations": relations_out,
                "audit-trail": audit_trail
            }
            
    json_str = json.dumps(final_data, ensure_ascii=False, indent=2)
    json_str = re.sub(r'"__find_symbol_\d+__":', '"find_symbol":', json_str)
    json_str = re.sub(r'"__extract_symbol_name_\d+__":', '"extract_symbol_name":', json_str)
    json_str = re.sub(r'"__extract_relation_\d+__":', '"extract_relation":', json_str)
    
    output_path = os.path.join(base_dir, 'dataset.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print(f"{output_path} created successfully!")

    return final_data

if __name__ == "__main__":
    main()
