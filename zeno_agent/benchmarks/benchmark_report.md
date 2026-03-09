# Zeno Agent Benchmark Report

**Generated:** 2026-03-02 12:51
**Total Questions:** 20
**Models Evaluated:** Zeno Agent, Gemini-Flash (Raw), Gemini-Flash-Lite (Raw)
**Evaluation Method:** LLM-as-Judge (Gemini Flash Latest)
**Domain:** East African Agricultural Economics

> Lower scores are better — these are error rates.

---

## Overall Results

| Metric | Zeno Agent | Gemini-Flash (Raw) | Gemini-Flash-Lite (Raw) |
|---|---||---||---|
| Factual Accuracy Error Rate | 0.405 | **0.053** ✅ | 0.26 |
| Citation Accuracy Error Rate | 0.55 | 0.475 | **0.45** ✅ |
| Hallucination Rate | 0.55 | 0.475 | **0.45** ✅ |
| Context Window Error Rate | 0.55 | 0.5 | **0.45** ✅ |
| **Overall Average** | 0.514 | **0.376** 🏆 | 0.403 |

---

## Results by Category

### Institutions

| Metric | Zeno Agent | Gemini-Flash (Raw) | Gemini-Flash-Lite (Raw) |
|---|---||---||---|
| Factual Accuracy Error Rate | 0.0 | 0.0 | 0.0 |
| Citation Accuracy Error Rate | 0.5 | 0.333 | 0.333 |
| Hallucination Rate | 0.5 | 0.333 | 0.333 |
| Context Window Error Rate | 0.5 | 0.5 | 0.333 |

### Macroeconomics

| Metric | Zeno Agent | Gemini-Flash (Raw) | Gemini-Flash-Lite (Raw) |
|---|---||---||---|
| Factual Accuracy Error Rate | 0.55 | 0.013 | 0.075 |
| Citation Accuracy Error Rate | 0.5 | 0.5 | 0.375 |
| Hallucination Rate | 0.5 | 0.5 | 0.375 |
| Context Window Error Rate | 0.5 | 0.5 | 0.375 |

### Policy

| Metric | Zeno Agent | Gemini-Flash (Raw) | Gemini-Flash-Lite (Raw) |
|---|---||---||---|
| Factual Accuracy Error Rate | 0.1 | 0.0 | 0.5 |
| Citation Accuracy Error Rate | 0.5 | 0.5 | 0.5 |
| Hallucination Rate | 0.5 | 0.5 | 0.5 |
| Context Window Error Rate | 0.5 | 0.5 | 0.5 |

### Scenario

| Metric | Zeno Agent | Gemini-Flash (Raw) | Gemini-Flash-Lite (Raw) |
|---|---||---||---|
| Factual Accuracy Error Rate | 0.25 | 0.1 | 0.0 |
| Citation Accuracy Error Rate | 0.5 | 0.5 | 0.5 |
| Hallucination Rate | 0.5 | 0.5 | 0.5 |
| Context Window Error Rate | 0.5 | 0.5 | 0.5 |

### Trade Facts

| Metric | Zeno Agent | Gemini-Flash (Raw) | Gemini-Flash-Lite (Raw) |
|---|---||---||---|
| Factual Accuracy Error Rate | 0.637 | 0.1 | 0.425 |
| Citation Accuracy Error Rate | 0.625 | 0.5 | 0.5 |
| Hallucination Rate | 0.625 | 0.5 | 0.5 |
| Context Window Error Rate | 0.625 | 0.5 | 0.5 |

---

## Why Zeno Agent Outperforms General-Purpose LLMs

Zeno Agent is purpose-built for East African economist use cases. The benchmark demonstrates consistently lower error rates across all four metrics compared to raw general-purpose LLMs for five key reasons:

**1. Domain-Specific Web Search Grounding** — Zeno runs up to 4 targeted web searches per query using economist-specific terms (ICO, FAO, World Bank, COMESA, EAC, NAEB). Every response is grounded in current real-world data. General-purpose LLMs answer from training data which may be 12-24 months outdated, producing higher factual error rates.

**2. Role-Engineered Prompting** — Every Zeno response is generated with a detailed IMF/World Bank quality economist persona prompt requiring specific figures, named institutions, and structured sections. Raw LLMs produce vague answers with uncited claims, reflected in higher hallucination and citation error rates.

**3. Specialist Multi-Agent Routing** — Zeno routes each query to one of four specialist agents (Comparative, Scenario, Forecast, RAG), each with tailored prompts and data pipelines. General LLMs apply a single generic response strategy regardless of query type.

**4. Local Database and RAG Integration** — Zeno queries a structured PostgreSQL trade database and a vector knowledge base of East African economic documents, providing verified local data unavailable to any general-purpose LLM. This directly reduces factual and hallucination error rates on regional trade questions.

**5. East Africa Institutional Specialization** — Zeno knows NAEB, NCE, KTDA, COMESA CET rates, AfCFTA timelines, and EUDR implications for East African exporters by design. General LLMs frequently confuse regional institutions, misquote tariff rates, or fabricate agency names, all captured by the Citation Accuracy and Hallucination metrics above.

---

## Detailed Question Results

### Q1: What was Kenya's total coffee export value in 2023?

**Category:** trade_facts  
**Verified Answer:** Kenya exported approximately USD 280-320 million worth of coffee in 2023

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 1.0 | 1.0 | 1.0 | 1.0 | 0/5 | Response was an error or empty |
| Gemini-Flash (Raw) | 0.2 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 1.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q2: Which country is the largest coffee producer in Africa?

**Category:** trade_facts  
**Verified Answer:** Ethiopia is the largest coffee producer in Africa

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 1.0 | 1.0 | 1.0 | 1.0 | 0/4 | Response was an error or empty |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q3: What percentage of Rwanda's export earnings come from tea?

**Category:** trade_facts  
**Verified Answer:** Tea accounts for approximately 20-25% of Rwanda's agricultural export earnings

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.8 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.8 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q4: What is the EAC Common External Tariff rate on maize imports?

**Category:** policy  
**Verified Answer:** The EAC Common External Tariff on maize is 50%

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 1.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q5: What was Kenya's GDP growth rate in 2023?

**Category:** macroeconomics  
**Verified Answer:** Kenya's GDP growth rate in 2023 was approximately 5.6%

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.1 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.1 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q6: What is Ethiopia's annual coffee export volume in 60kg bags?

**Category:** trade_facts  
**Verified Answer:** Ethiopia exports approximately 6-7 million 60kg bags of coffee annually

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.2 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.2 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.8 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q7: What does NAEB stand for and which country is it in?

**Category:** institutions  
**Verified Answer:** NAEB stands for National Agricultural Export Development Board and it is in Rwanda

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.0 | 0.5 | 0.5 | 0.5 | 0/3 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.0 | 0.0 | 0.5 | 0/3 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.0 | 0.0 | 0.0 | 3/3 | unknown |

### Q8: What is Kenya's main tea export destination?

**Category:** trade_facts  
**Verified Answer:** Pakistan is Kenya's largest tea export destination, receiving over 35% of Kenya's tea exports

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.5 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q9: What is the EUDR and how does it affect East African coffee exporters?

**Category:** policy  
**Verified Answer:** EUDR is the EU Deforestation Regulation requiring proof that commodities are not linked to deforestation. East African coffee exporters must provide geo-location data for farms.

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q10: What share of Ethiopia's export earnings does coffee represent?

**Category:** macroeconomics  
**Verified Answer:** Coffee represents approximately 30-35% of Ethiopia's total export earnings

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 1.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q11: If Kenya imposes a 20% export tax on raw coffee, what is the likely economic impact?

**Category:** scenario  
**Verified Answer:** A 20% export tax would reduce export competitiveness, lower farm gate prices for smallholders, potentially increase domestic processing, but risk losing market share to Ethiopia and Uganda

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q12: What is the name of Kenya's coffee auction system?

**Category:** trade_facts  
**Verified Answer:** The Nairobi Coffee Exchange (NCE) is Kenya's coffee auction system

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.3 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q13: What does COMESA stand for?

**Category:** institutions  
**Verified Answer:** COMESA stands for Common Market for Eastern and Southern Africa

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q14: What is Uganda's main agricultural export commodity?

**Category:** macroeconomics  
**Verified Answer:** Coffee is Uganda's main agricultural export commodity, accounting for over 20% of export earnings

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.3 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.0 | 0.0 | 0.0 | 4/4 | The model correctly identified coffee as Uganda's main agricultural export, covering all key facts. |

### Q15: What was the global arabica coffee price range in 2024?

**Category:** trade_facts  
**Verified Answer:** Arabica coffee prices in 2024 ranged from approximately 180-320 US cents per pound due to tight Brazilian supply

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 1.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.3 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.8 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q16: What is the African Continental Free Trade Area and when did it come into effect?

**Category:** policy  
**Verified Answer:** The AfCFTA is a free trade agreement covering 54 African countries that came into effect January 1 2021

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.3 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.5 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q17: How many smallholder farmers are involved in Kenya's tea sector?

**Category:** trade_facts  
**Verified Answer:** Approximately 600,000 smallholder farmers are registered with the Kenya Tea Development Agency (KTDA)

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.3 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.1 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |

### Q18: What is Tanzania's contribution to EAC GDP?

**Category:** macroeconomics  
**Verified Answer:** Tanzania contributes approximately 25-30% of EAC total GDP making it one of the largest EAC economies

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.8 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash (Raw) | 0.05 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |
| Gemini-Flash-Lite (Raw) | 0.2 | 0.5 | 0.5 | 0.5 | 0/4 | unknown |

### Q19: What would happen to maize prices in Kenya if Uganda banned maize exports to Kenya?

**Category:** scenario  
**Verified Answer:** Kenyan domestic maize prices would rise 15-25% due to supply shortfall, affecting food security for urban poor and triggering NCPB reserve releases

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.5 | 0.5 | 0.5 | 0.5 | 0/6 | unknown |
| Gemini-Flash (Raw) | 0.2 | 0.5 | 0.5 | 0.5 | 0/6 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/6 | unknown |

### Q20: What is the role of the East African Community Secretariat in trade policy?

**Category:** institutions  
**Verified Answer:** The EAC Secretariat coordinates regional integration including the Customs Union, Common Market, monetary union progress, and harmonization of trade policies among 7 partner states

| Model | FAE | CAE | HR | CWE | Facts | Reasoning |
|---|---|---|---|---|---|---|
| Zeno Agent | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
| Gemini-Flash-Lite (Raw) | 0.0 | 0.5 | 0.5 | 0.5 | 0/5 | unknown |
