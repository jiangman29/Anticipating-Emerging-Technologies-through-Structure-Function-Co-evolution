import os
import json
import time
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 🚀 设置 OpenAI API Key ==========

DEEPSEEK_API_URL = "https://XXXXX"

# ========== 🚀 读取 Excel 数据 ==========
def load_patent_data(filepath):
    df = pd.read_excel(filepath)
    df = df[["id", "Structure Part A", "Structure Part B", "Function Part"]]
    df.fillna('', inplace=True)
    return df.to_dict(orient="records")

# ========== 🚀 1. 生成用于 DeepSeek 处理的 Chain of Thought (CoT) Prompt ==========
def generate_prompt(structure_part_a, structure_part_b, function_part):
    prompt = f"""
### Instruction:
You are an expert patent analyst specializing in digital medical technology patents. Your task is to:
1. **Extract Technical Structure (S) and Technical Function (F)** from the given patent text.
   - **Technical Structure (S)**: Refers to terms or phrases describing technological components, systems, devices, or modules.
   - **Technical Function (F)**: Refers to terms or phrases describing the application scenarios, technical effects, advantages, or improvements of a technology.
2. **Identify relationships between these elements:**
   - **S-S Relationship**: Structural hierarchy or semantic relationships between technical structures.
   - **S-F Relationship**: Functional dependency between structures and functions.

### Step 1: Named Entity Recognition (NER)
#### 1. Extract **Technical Structure (S)**: 
  - Prioritize extraction from "Structure Part A".
  - Extract the first technical term from the first sentence as the primary structure S_main
      - If Structure Part A has only one sentence, ensure S_main is the core technology phrase (e.g., extract “early warning system” from “Pig raising site environment monitoring control and disease prediction early warning system…”).
      - Avoid generic terms (e.g., "system", "device", "unit").
      - If no valid term exists in the first sentence, do not assign S_main.
  - Extract from "Structure Part B" for additional structures, but prioritize "Structure Part A" entities.
  - Keep core technology components or system; avoid generic terms (e.g., "system", "device"); avoid overly long expressions (e.g., excluding S_main, each extracted S should not exceed 5 words).

#### 2. Extract **Technical Function (F)**:
  - Extract only from "Function Part".
  - If multiple functional scenarios are listed, keep only those relevant to digital medical technology (e.g., telemedicine, remote monitoring, wearable devices).
  - Apply lemmatization to normalize function terms.
  - Retain key functional terms, removing redundant descriptors.
  
---  
### **Important Notes: Think carefully before Before Relationship Extraction.**
- **Ensure that the extracted S and F entities are correct before identifying relationships.**
- **Do not generate relationships until all S and F extractions are validated.**
- **Ensure extracted terms are unique within each patent.** 
---

### Step 2: Extract Relationships (RE)
#### 1. **S-S Relationship**:
  - Identify hierarchical relations from **Structure Part A**: Use S_main as the primary node, linking it to all other S entities.
  - Supplement with relations from **Structure Part B**: if multiple S entities appear in the same sentence and form a semantic connection.
  - Ensure that both structures in an S-S pair are distinct.
  
#### 2. **S-F Relationship**:
  - Link **S_main** to all extracted F entities from **Function Part**.
  - Supplement S-F relationships from **Function Part**: if S and F appear in the same sentence and exhibit a clear functional dependency, create an S-F relationship.
  - Ensure consistency by matching F terms in their base form when applicable.

---
###Important Notes on Output:
- Deduplication: Ensure that there are no duplicate entities within the same patent text (keep only one instance).
- Empty Arrays: If no eligible entities or relationships are identified, return an empty array for the corresponding field.
---

### Input Patent Text
#### **Structure Part A:**
{structure_part_a}

#### **Structure Part B:**
{structure_part_b}

#### **Function Part:**
{function_part}

---
### Response Format (strict JSON, no explanation)
```json
{{
  "Structure": ["entity1", "entity2", ...],
  "Function": ["function1", "function2", ...],
  "S-S": [
    {{"Structure 1": "entity1", "Structure 2": "entity2"}},
    {{"Structure 1": "entity3", "Structure 2": "entity4"}}
  ],
  "S-F": [
    {{"Structure": "entity1", "Function": "function1"}},
    {{"Structure": "entity2", "Function": "function2"}}
  ]
}}
```

---Examples
### **Example 1 (Standard Structure)**
#### 📌 **Input Text**
##### **Structure Part A**
the medicament dispensing monitoring system comprises Medicament dispensing station. the medicament dispensing monitoring system comprises Monitoring unit. the medicament dispensing monitoring system comprises Medicament dispensing station. the medicament dispensing monitoring system comprises Identification unit. the medicament dispensing monitoring system comprises Releasable locking mechanism.

##### **Structure Part B**
The method involves capturing respective identifiers relating to the individual attempting to dispense medications from the medicament dispensing station. The time at which the individual attempts to dispense the medicaments from the medicament dispensing station is captured. The captured and identifiers and time at which the individual attempts to dispense the medicaments is transmitted to the remote monitoring unit over communication network. The captured identifiers and time are stored in remote monitoring memory for future reference.

##### **Function Part**
Method for monitoring dispensing of medicaments in hospital from portable medicament dispensing station. The tampering with the data input to the system is obviated and traceability and accountability are facilitated effectively.

#### ✅ **Output JSON**
```json
{{
  "Structure": ["medicament dispensing monitoring system", "medicament dispensing station", "monitoring unit", "identification unit", "releasable locking mechanism", "communication network", "remote monitoring memory"],
  "Function": ["obviated tampering", "monitoring dispensing", "traceability", "accountability"],
  "S-S": [
    {{"Structure 1": "medicament dispensing monitoring system", "Structure 2": "medicament dispensing station"}},
    {{"Structure 1": "medicament dispensing monitoring system", "Structure 2": "monitoring unit"}},
    {{"Structure 1": "medicament dispensing monitoring system", "Structure 2": "identification unit"}},
    {{"Structure 1": "medicament dispensing monitoring system", "Structure 2": "releasable locking mechanism"}},
    {{"Structure 1": "monitoring unit", "Structure 2": "communication network"}}
  ],
  "S-F": [
    {{"Structure": "medicament dispensing monitoring system", "Function": "obviated tampering"}},
	{{"Structure": "medicament dispensing monitoring system", "Function": "monitoring dispensing"}},
	{{"Structure": "medicament dispensing monitoring system", "Function": "traceability"}},
    {{"Structure": "medicament dispensing monitoring system", "Function": "accountability"}},
    {{"Structure": "medicament dispensing station", "Function": "capturing respective identifiers"}}
  ]
}}

### **Example 2（"Structure Part A" only one sentence）**
#### 📌 **Input Text**
##### **Structure Part A**
a medical system construction method

##### **Structure Part B**
The method involves transmitting first service data by a hospital information system. Second service data is received by using a hospital resource plan system. Service process analysis function is determined according to the first and second service data. Relevant characteristics of the first and second service data are detected according to basic medical service process. The medical service process is performed according to an integrated hospital information system. The hospital information system is provided with the hospital resource plan system to detect integrated management function.

##### **Function Part**
Medical system construction method. The method enables ensuring better real time updating effect and improving hospital working efficiency.

#### ✅ **Output JSON**
```json
{{
  "Structure": ["medical system", "hospital information system", "hospital resource plan system", "service process analysis function", "medical service process", "integrated hospital information system", "integrated management function"],
  "Function": ["real time updating effect", "hospital working efficiency"],
  "S-S": [
    {{"Structure 1": "hospital information system", "Structure 2": "service process analysis function"}},
    {{"Structure 1": "hospital resource plan system", "Structure 2": "service process analysis function"}},
    {{"Structure 1": "medical service process", "Structure 2": "integrated hospital information system"}},
    {{"Structure 1": "hospital information system", "Structure 2": "hospital resource plan system"}},
    {{"Structure 1": "medical system", "Structure 2": "hospital resource plan system"}},
    {{"Structure 1": "medical system", "Structure 2": "integrated management function"}},
    {{"Structure 1": "medical system", "Structure 2": "hospital information system"}},
    {{"Structure 1": "medical system", "Structure 2": "service process analysis function"}},
    {{"Structure 1": "medical system", "Structure 2": "medical service process"}},
    {{"Structure 1": "medical system", "Structure 2": "integrated hospital information system"}}
],
  "S-F": [
    {{"Structure": "medical system", "Function": "real time updating effect"}},
    {{"Structure": "medical system", "Function": "hospital working efficiency"}}
  ]
}}

### **Example 3（"Structure Part A" empty）**
#### 📌 **Input Text**
##### **Structure Part A**


##### **Structure Part B**
A control system for head and neck cancer radiotherapy production method, involves reading or entering patients record in electronic device, setting position for patient based on patients condition, detecting surface temperature of head and neck with detector, where the surface temperature information is transmitted to the host computer, displaying suggested body location information on real-time display, followed by setting an alarm module when abnormal situation occurs, and sending remote diagnosis request information through telemedicine terminal.

##### **Function Part**
Method for producing control system for head and neck cancer radiotherapy.

#### ✅ **Output JSON**
```json
{{
  "Structure": ["control system", "electronic device", "detector", "host computer", "real-time display", "alarm module", "telemedicine terminal"],
  "Function": ["head and neck cancer radiotherapy"],
  "S-S": [
    {{"Structure 1": "control system", "Structure 2": "electronic device"}},
    {{"Structure 1": "control system", "Structure 2": "detector"}},
    {{"Structure 1": "control system", "Structure 2": "host computer"}},
    {{"Structure 1": "control system", "Structure 2": "real-time display"}},
    {{"Structure 1": "control system", "Structure 2": "alarm module"}},
    {{"Structure 1": "control system", "Structure 2": "telemedicine terminal"}}
],
  "S-F": [
    {{"Structure": "control system", "Function": "head and neck cancer radiotherapy"}}
  ]
}}
"""
    return prompt


# ========== 2. 调用 DeepSeek API ==========
def call_deepseek(prompt, api_key, retries=5):  #也可以3次重试机会
    """ 调用 DeepSeek API 并解析 JSON """
    headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",  #可以更改 DeepSeek-V3/DeepSeek-R1/DeepSeek-V3-0324/DeepSeek-V3-0324-fast
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2    # 低温度提高稳定性
    }

    for attempt in range(retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            raw_content = response.json()["choices"][0]["message"]["content"]
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_content)
        except Exception as e:
            print(f"[{api_key[:10]}...] 第{attempt+1}次调用失败：{e}")
            time.sleep(5)
    return None


# ========== 3. 处理单条专利 ==========
def process_single_patent(patent, api_key):
    patent_id = patent.get("id", "Unknown")
    structure_part_a = patent.get("Structure Part A", "")
    structure_part_b = patent.get("Structure Part B", "")
    function_part = patent.get("Function Part", "")
    time.sleep(2)
    prompt = generate_prompt(structure_part_a, structure_part_b, function_part)
    result = call_deepseek(prompt, api_key)
    if result and isinstance(result, dict):
        result["Patent ID"] = patent_id
        return result
    else:
        return {"Patent ID": patent_id, "Error": True}


# ========== 🧰 单 Key 线程任务 ==========
def run_worker(patent_batch, api_key, output_prefix):
    results = []
    failed = []
    for patent in tqdm(patent_batch, desc=f"🔑 Key {api_key[:10]}..."):
        res = process_single_patent(patent, api_key)
        if res and "Error" not in res:
            results.append(res)
        else:
            failed.append(patent)

    os.makedirs("result", exist_ok=True)
    with open(f"result/{output_prefix}_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(f"result/{output_prefix}_failed.json", "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    print(f"✅ Key {api_key[:10]}... 处理完成，成功 {len(results)} 条，失败 {len(failed)} 条")


if __name__ == "__main__":
    # 你准备的 5 个文件名（完整路径）：
    batch_files = [
        "batches/batch1.xlsx",
        "batches/batch2.xlsx",
        "batches/batch3.xlsx",
        "batches/batch4.xlsx",
        "batches/batch5.xlsx",
    ]

    # 对应的 5 个 API Keys
    api_keys = [
        "sk-1XXXXX",
        "sk-2XXXX",
        "sk-3XXXXX",
        "sk-4XXXXX",
        "sk-5XXXXXX"
    ]

    # 每个 key 对应一个文件：并行处理（加快速度）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        for filepath, key in zip(batch_files, api_keys):
            executor.submit(run_worker, load_patent_data(filepath), key, os.path.splitext(os.path.basename(filepath))[0])
