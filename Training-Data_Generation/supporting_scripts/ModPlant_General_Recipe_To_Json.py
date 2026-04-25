import xml.etree.ElementTree as ET
import json

def parse_general_recipe(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {'b2mml': 'http://www.mesa.org/xml/B2MML'}

    def txt(parent, xpath):
        """Safe text getter: returns None if parent or child is missing."""
        if parent is None:
            return None
        node = parent.find(xpath, ns)
        return node.text if (node is not None and node.text is not None) else None

    recipe_data = {
        'ID': txt(root, 'b2mml:ID'),
        'Description': txt(root, 'b2mml:Description'),
        'Inputs': [],
        'Outputs': [],
        'Intermediates': [],
        'ProcessElements': [],
        'DirectedLinks': []
    }

    # -------- Process Inputs --------
    for material in root.findall('.//b2mml:ProcessInputs/b2mml:Material', ns):
        amt = material.find('b2mml:Amount', ns)  # anchor to Amount first
        recipe_data['Inputs'].append({
            'ID':           txt(material, 'b2mml:ID'),
            'Description':  txt(material, 'b2mml:Description'),
            'Quantity':     txt(amt, 'b2mml:QuantityString'),
            'DataType':     txt(amt, 'b2mml:DataType'),
            'UnitOfMeasure':txt(amt, 'b2mml:UnitOfMeasure'),
            'Key':          txt(amt, 'b2mml:Key')
        })

    # -------- Process Outputs --------
    for material in root.findall('.//b2mml:ProcessOutputs/b2mml:Material', ns):
        amt = material.find('b2mml:Amount', ns)
        recipe_data['Outputs'].append({
            'ID':           txt(material, 'b2mml:ID'),
            'Description':  txt(material, 'b2mml:Description'),
            'Quantity':     txt(amt, 'b2mml:QuantityString'),
            'DataType':     txt(amt, 'b2mml:DataType'),
            'UnitOfMeasure':txt(amt, 'b2mml:UnitOfMeasure'),
            'Key':          txt(amt, 'b2mml:Key')
        })

    # -------- Process Intermediates --------
    for material in root.findall('.//b2mml:ProcessIntermediates/b2mml:Material', ns):
        amt = material.find('b2mml:Amount', ns)
        recipe_data['Intermediates'].append({
            'ID':           txt(material, 'b2mml:ID'),
            'Description':  txt(material, 'b2mml:Description'),
            'Quantity':     txt(amt, 'b2mml:QuantityString'),
            'DataType':     txt(amt, 'b2mml:DataType'),
            'UnitOfMeasure':txt(amt, 'b2mml:UnitOfMeasure'),
            'Key':          txt(amt, 'b2mml:Key')
        })

    # -------- Directed Links (Workflow) --------
    for link in root.findall('.//b2mml:DirectedLink', ns):
        recipe_data['DirectedLinks'].append({
            'ID':     txt(link, 'b2mml:ID'),
            'FromID': txt(link, 'b2mml:FromID'),
            'ToID':   txt(link, 'b2mml:ToID')
        })

    # -------- Process Elements (Steps) --------
    for process_element in root.findall('.//b2mml:ProcessElement', ns):
        pe_data = {
            'ID':          txt(process_element, 'b2mml:ID'),
            'Description': txt(process_element, 'b2mml:Description'),
            'Parameters': [],
            'SemanticDescription': None
        }

        # Only search immediate children 'ProcessElementParameter' below this PE
        for param in process_element.findall('b2mml:ProcessElementParameter', ns):
            val = param.find('b2mml:Value', ns)  # anchor to Value first
            pe_data['Parameters'].append({
                'ID':           txt(param, 'b2mml:ID'),
                'Description':  txt(param, 'b2mml:Description'),
                'ValueString':  txt(val, 'b2mml:ValueString'),
                'DataType':     txt(val, 'b2mml:DataType'),
                'UnitOfMeasure':txt(val, 'b2mml:UnitOfMeasure'),
                'Key':          txt(val, 'b2mml:Key')
            })

        # Semantic Description (optional)
        sem_val = process_element.find('.//b2mml:OtherInformation/b2mml:OtherValue', ns)
        pe_data['SemanticDescription'] = txt(sem_val, 'b2mml:ValueString')

        recipe_data['ProcessElements'].append(pe_data)

    return recipe_data

import os
import re
import json
from datetime import datetime

def save_parsed_recipe_json_by_id(xml_path: str, 
                                  out_dir: str | None = None, 
                                  with_timestamp: bool = False) -> str:
    """
    Parse the GeneralRecipe XML and save the JSON to the Recipe/Json directory.
    
    Args:
        xml_path: Path to the GeneralRecipe XML.
        out_dir:  Optional override. If None, defaults to <ScriptDir>/Recipe/Json.
        with_timestamp: If True, append _YYYYMMDD_HHMMSS to the file name.
        
    Returns:
        Absolute path to the written JSON file.
    """
    
    # 1. Parse to dict (assuming parse_general_recipe is defined elsewhere)
    data = parse_general_recipe(xml_path)

    # 2. Get <ID> and make it filename-safe
    rid = (data.get("ID") or "GeneralRecipe").strip()
    rid_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", rid).strip("_") or "GeneralRecipe"

    # 3. Build output filename
    if with_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{rid_slug}_parsed_{ts}.json"
    else:
        fname = f"{rid_slug}_parsed.json"

    # 4. Determine Target Directory
    if out_dir:
        # If user provides a specific path, use it
        target_dir = out_dir
    else:
        # Default: Use "Recipe/Json" relative to this script file
        # Check if __file__ is available (works in .py scripts)
        try:
            current_base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for Jupyter Notebooks where __file__ might not exist
            current_base = os.getcwd()
            
        # Join paths: e.g., .../YourProject/Recipe/Json
        target_dir = os.path.join(current_base, "Recipe", "Json")

    # 5. Create directory if it doesn't exist (handles missing "Recipe" or "Json" folder)
    os.makedirs(target_dir, exist_ok=True)
    
    # 6. Combine full path
    out_path = os.path.join(target_dir, fname)

    # 7. Write JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved JSON to: {out_path}")
    return out_path


# Example usage:
# data = parse_general_recipe("ExampleGeneralRecipe.xml")
# print(json.dumps(data, indent=4, ensure_ascii=False))
# with open("parsed_recipe_output.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)
