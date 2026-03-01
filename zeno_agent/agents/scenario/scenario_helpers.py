from zeno_agent.embedding_utils import encode_query_to_vector
from zeno_agent.db_utils import query_rag_embeddings_semantic


def merge_rag_content(rag_results):
    seen = set()
    merged = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        merged.append(content)
    return " ".join(merged)


def build_scenario_prompt(query, structured_context, rag_context):
    return f"""You are Dr. Zeno, Senior Economist at the East African Trade Institute. You are producing a policy impact assessment brief for a Minister of Agriculture and a World Bank country economist.

SCENARIO QUERY:
{query}

LOCAL DATA:
- Structured Economic Data: {structured_context}
- Knowledge Base Documents: {rag_context}

Write a comprehensive economic scenario analysis with the following sections as flowing paragraphs:

SECTION 1 - BASELINE CONDITIONS
Current state of the commodity and sector: production volumes, prices, farmer income, export performance, and government spending context. Include specific figures.

SECTION 2 - IMMEDIATE FIRST-ORDER EFFECTS (Year 1)
Quantify direct impacts: output increase, price changes, fiscal cost, winners and losers. Apply supply elasticity theory with real coefficients.

SECTION 3 - PRICE DYNAMICS AND MARKET EQUILIBRIUM
Supply curve shift analysis, domestic price impact, export price effects, consumer surplus changes, and price transmission to neighboring countries.

SECTION 4 - TRADE AND EXPORT EFFECTS
Changes in export volumes, foreign exchange earnings, market share in key destinations, and competitiveness versus rival producers.

SECTION 5 - MACROECONOMIC IMPLICATIONS
GDP contribution, fiscal multiplier, employment generation, currency effects, inflation transmission, balance of payments impact.

SECTION 6 - SECOND-ORDER RISKS AND UNINTENDED CONSEQUENCES
Rent-seeking risks, market distortion, WTO compliance, environmental sustainability, subsidy dependency trap, political economy constraints.

SECTION 7 - HISTORICAL COMPARISONS
Two or three similar policy interventions in Africa or globally with specific outcomes — what worked, what failed, and why.

SECTION 8 - RECOMMENDATIONS AND DESIGN PRINCIPLES
Specific design recommendations: targeting criteria, delivery mechanism, duration, monitoring KPIs, exit strategy, and budget envelope. What NOT to do based on evidence.

WRITING STANDARDS:
- IMF Article IV report quality — specific data, real institutions, policy names, dollar figures
- No bullet points, no markdown headers — flowing professional paragraphs
- 800 to 1000 words total
- End with one concrete recommendation the Minister should implement in the next 90 days

BEGIN POLICY IMPACT ASSESSMENT:""".strip()