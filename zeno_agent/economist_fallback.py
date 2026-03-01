import os
from dotenv import load_dotenv
from google import genai
from .web_search import search_web, format_web_results

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
GENERATION_MODEL = "models/gemini-2.5-flash"


def economist_web_answer(
    query: str,
    agent_type: str,
    commodity: str = "",
    country: str = "",
    extra_context: str = ""
) -> str:
    """
    Called by any sub-agent when DB has no data or as primary answer engine.
    Searches web with multiple targeted queries and generates IMF/World Bank
    quality economist answer.
    agent_type: "comparative" | "scenario" | "forecast" | "rag"
    """

    if agent_type == "comparative":
        search_queries = [
            f"{commodity} export statistics {country} annual data volumes value 2023 2024 2025",
            f"{commodity} trade competitiveness East Africa market share global benchmark",
            f"{commodity} export policy {country} government strategy incentives",
            f"{commodity} price trends {country} comparison world market ICO FAO",
            f"East Africa {commodity} trade report COMESA EAC World Bank IMF 2024",
        ]
    elif agent_type == "scenario":
        search_queries = [
            f"{country} {commodity} production statistics GDP contribution farmers 2024 2025",
            f"{commodity} subsidy policy Africa outcomes effects case study evidence",
            f"{country} {commodity} export value volume market 2023 2024",
            f"{commodity} global price trends supply demand 2025 outlook",
            f"{country} agricultural policy {commodity} government intervention World Bank IMF",
        ]
    elif agent_type == "forecast":
        search_queries = [
            f"{commodity} price forecast 2025 2026 outlook {country} projections",
            f"{commodity} global supply demand balance 2025 USDA FAO ICO",
            f"{commodity} export market trends East Africa 2025 2026",
            f"{country} {commodity} production yield forecast weather seasonal",
            f"ICO FAO World Bank {commodity} commodity price outlook report 2025",
        ]
    else:
        search_queries = [
            f"{query} East Africa economic data statistics 2024 2025",
            f"{query} Kenya Tanzania Uganda Ethiopia trade policy analysis",
            f"{query} World Bank IMF FAO African Development Bank report",
            f"{query} policy implications Sub-Saharan Africa economist analysis",
        ]

    # Run all search queries, collect unique results
    all_web_results = []
    seen_links = set()
    for sq in search_queries[:4]:
        results = search_web(query=sq, num_results=3)
        for r in results:
            if r["link"] not in seen_links and r.get("snippet", "").strip():
                seen_links.add(r["link"])
                all_web_results.append(r)

    web_content = format_web_results(all_web_results)
    has_web = bool(web_content)

    print(f"[EconomistFallback] agent={agent_type} | commodity={commodity} | country={country} | web_results={len(all_web_results)}")

    # ── Role and structure per agent type ──────────────────────────────────

    if agent_type == "comparative":
        role_brief = "comparative trade economist producing a policy brief for the East African Community Secretariat"
        structure = """
Structure your analysis with the following clearly labelled sections written as flowing paragraphs. Each section must be at least 150 words.

PRODUCTION AND EXPORT OVERVIEW
Compare historical production volumes in metric tonnes, export values in USD millions, global and regional market share percentages, number of smallholder farmers involved, contribution to GDP and foreign exchange earnings for each country. Use every specific figure from the web results and supplement with your expert knowledge.

PRICE DYNAMICS AND MARKET POSITIONING
Compare average FOB export prices per kg or per tonne, quality premiums or discounts relative to global benchmarks, auction versus direct export systems, organic and Fair Trade certification rates, specialty market penetration, and buyer concentration by destination country.

COMPETITIVE STRENGTHS AND WEAKNESSES
Analyze cost of production per kg, yield per hectare versus global average, post-harvest processing infrastructure quality, port and logistics costs, government extension services, cooperative membership rates, and export destination diversification index.

TRADE POLICY AND INSTITUTIONAL FRAMEWORK
Compare export taxes and levies as percentage of FOB value, marketing board roles, cooperative law frameworks, phytosanitary certification systems, compliance with EU Deforestation Regulation (EUDR), WTO commitments, and bilateral preferential agreements.

GROWTH TRAJECTORY AND HISTORICAL PERFORMANCE
Provide compound annual growth rates for export volumes and values over the last 5 to 10 years. Identify inflection points caused by policy changes, weather events, or global price cycles. Compare projected growth under current trajectory.

RISKS AND STRUCTURAL BLIND SPOTS
Cover climate change vulnerability including rainfall variability and temperature stress projections, coffee leaf rust or tea blister blight disease pressure, EUDR compliance costs for smallholders, currency depreciation impact on farmer net income, youth migration away from agriculture, and land tenure insecurity.

POLICY RECOMMENDATIONS
Provide five specific actionable recommendations per country. Each recommendation must name the responsible institution, the budget envelope or funding source, the implementation timeline in months, and the measurable outcome indicator. Include at least one recommendation on value chain upgrading, one on market diversification, and one on climate resilience."""

    elif agent_type == "scenario":
        role_brief = "policy impact economist advising the East African Community and the World Bank on agricultural intervention design"
        structure = """
Structure your analysis with the following clearly labelled sections written as flowing professional paragraphs. Each section must be at least 150 words. Do not use bullet points or markdown.

BASELINE CONDITIONS
Describe the current state of the commodity sector in the specified country. Include: total production volume in metric tonnes, number of smallholder farmers, contribution to GDP as a percentage, annual export value in USD, average farm gate price, government budget allocation to the sector, and any existing subsidy or support programs. Use specific figures from the web results. State the year of each data point.

IMMEDIATE FIRST-ORDER EFFECTS — YEAR ONE
Quantify the direct impact of the proposed intervention. Estimate: the percentage increase in output using a supply price elasticity coefficient (cite the elasticity value used), the change in domestic price level, the fiscal cost of the subsidy as a percentage of agricultural GDP, the number of farmers who would benefit directly, and the likely change in export volume in year one.

PRICE DYNAMICS AND MARKET EQUILIBRIUM
Analyze how the subsidy shifts the domestic supply curve using partial equilibrium theory. Estimate the producer surplus gain and consumer surplus effect. Assess whether increased supply would depress domestic prices and by how much, using price transmission elasticity estimates. Discuss export price effects and whether the country is a price taker or price maker in the global market for this commodity. Analyze spillover price transmission to neighboring countries.

TRADE AND EXPORT EFFECTS
Project changes in annual export volumes and FOB export values in USD over a three year horizon. Identify the top five importing countries for this commodity from the specified country and assess their likely demand response. Analyze whether the subsidy improves or worsens competitiveness relative to rival producers such as India for tea, Vietnam for coffee, or Argentina for soy. Assess foreign exchange earnings impact.

MACROECONOMIC IMPLICATIONS
Calculate the fiscal multiplier effect of the subsidy injection into the rural economy. Estimate total employment generation including direct farm jobs and indirect processing and logistics employment. Assess the balance of payments impact from increased export earnings. Analyze inflation transmission risk if domestic prices are affected. Estimate the GDP contribution change over a five year horizon.

SECOND-ORDER RISKS AND UNINTENDED CONSEQUENCES
Identify and quantify: rent-seeking and elite capture risk given land ownership concentration, market distortion effects on non-subsidized competing crops, WTO Agreement on Agriculture compliance limits on trade-distorting support measured by Aggregate Measure of Support, environmental sustainability risk from input intensification including fertilizer runoff and water use, subsidy dependency trap risk and fiscal sustainability beyond the program period, and political economy constraints on subsidy removal once established.

HISTORICAL POLICY COMPARISONS
Analyze three comparable subsidy interventions in Africa or Asia with specific outcomes. For each case provide: the country and commodity, the program design, the budget size, the measured output response, the fiscal cost per tonne produced, unintended consequences observed, and the policy lesson applicable to the current proposal. Examples to consider: Kenya fertilizer subsidy program, Ethiopia coffee sector reform, India tea board support schemes, Rwanda crop intensification program.

RECOMMENDATIONS AND PROGRAM DESIGN PRINCIPLES
Provide a complete program design framework including: the optimal subsidy instrument (input voucher, output price floor, or direct income transfer) with justification, targeting criteria to reach smallholders under two hectares, the delivery mechanism through cooperatives or digital mobile platforms, the recommended budget envelope as a percentage of agricultural GDP, duration of the program with a credible exit timeline, five key performance indicators with baseline values and targets, the monitoring and evaluation institution, and three things the government must NOT do based on the historical evidence above. End with a single headline recommendation the Minister should announce in the next 30 days."""

    elif agent_type == "forecast":
        role_brief = "commodity market economist producing a price outlook brief for institutional traders and government procurement officers"
        structure = """
Structure your analysis with the following clearly labelled sections written as flowing paragraphs. Each section must be at least 120 words.

CURRENT MARKET CONDITIONS
Describe the present price level in USD per tonne and local currency per kg, the recent 12-month price trend as a percentage change, the global supply-demand balance as surplus or deficit in metric tonnes, and key market-moving events of the past two quarters.

SUPPLY SIDE DRIVERS
Cover production forecasts from the top five producing countries in metric tonnes, current weather outlook including La Nina or El Nino status and its historical impact on yields, disease and pest pressure, fertilizer cost trends as a percentage of production cost, and planted acreage changes.

DEMAND SIDE DRIVERS
Analyze import demand from the EU, USA, China, and Middle East with year-on-year percentage changes, consumer preference shifts toward specialty or certified products, industrial demand from food processing and biofuel sectors, and sovereign buffer stock building behavior.

PRICE FORECAST WITH RANGES
Provide quarterly price forecasts for the next four quarters. For each quarter state: the base case price in USD and local currency, the bull case price and its trigger condition, the bear case price and its trigger condition, and the confidence level as a percentage based on historical forecast accuracy.

RISK FACTORS AND SCENARIO TRIGGERS
Identify the top five risks ranked by probability multiplied by impact. For each risk state the probability as a percentage, the price impact as a percentage change, and the earliest date it could materialize. Include geopolitical, weather, policy, and currency risks.

EAST AFRICA SPECIFIC IMPLICATIONS
Translate global forecasts into country-specific farm gate price expectations, export revenue projections in USD millions, food security risk assessment if the commodity is also a staple, and specific government buffer stock quantity recommendations in metric tonnes.

ACTIONABLE RECOMMENDATIONS
Provide specific recommendations for four stakeholder groups: government procurement officers on timing and volume of purchases, smallholder farmers on planting decisions and input investment, exporters and commodity traders on hedging strategy and contract timing, and development finance institutions on price risk facility design."""

    else:
        role_brief = "senior development economist at the East African Trade Institute"
        structure = """
Structure your analysis with the following clearly labelled sections written as flowing paragraphs. Each section must be at least 120 words.

EXECUTIVE SUMMARY
Summarize the key finding and most important recommendation in three to four sentences. State the most important number or statistic that defines the situation.

CONTEXT AND BACKGROUND
Historical context over the last ten years, current state of affairs with specific data points, institutional landscape, and why this topic is strategically important for East Africa now.

DETAILED ECONOMIC ANALYSIS
Comprehensive examination covering economic mechanisms with theory references, quantitative evidence from data, institutional and governance factors, regional integration dynamics, and comparison with peer countries or regions.

RISKS AND VULNERABILITIES
Key structural obstacles, external shocks, governance failures, and factors that could significantly worsen the situation. Quantify each risk where possible.

POLICY RECOMMENDATIONS
Five specific actionable recommendations. For each: name the responsible institution, the funding source, the implementation timeline, and the measurable success indicator."""

    # ── Build prompt ────────────────────────────────────────────────────────

    if has_web:
        prompt = f"""You are Dr. Zeno, {role_brief}. You are producing a professional economic policy brief that will be read by Ministers of Trade and Agriculture, central bank governors, IMF and World Bank country economists, and institutional commodity traders.

USER QUERY:
{query}

{f"LOCAL DATA CONTEXT (from internal database and knowledge base):{chr(10)}{extra_context}{chr(10)}" if extra_context else ""}

LIVE WEB RESEARCH (extract every specific fact, figure, statistic, percentage, date, institution name, and policy name):
{web_content}

{structure}

ABSOLUTE WRITING REQUIREMENTS:
- Minimum 1000 words. Maximum 1500 words. Never truncate a section.
- Extract and use every specific number from the web results — do not leave any statistic unused
- Write at World Bank Policy Research Paper or IMF Article IV Staff Report quality
- Name specific institutions: World Bank, FAO, IFAD, AfDB, COMESA, EAC, AGRA, USAID, bilateral donors
- Reference real programs by name: Rwanda Crop Intensification Program, Kenya Tea Development Agency, Ethiopian Coffee and Tea Authority
- Do NOT say "according to the web results" or "the search results indicate" — integrate facts naturally as a world-class economist would
- Do NOT use bullet points, numbered lists, or markdown formatting
- Do NOT use asterisks, hashes, or any special characters
- Write every section as flowing professional paragraphs only
- Every section heading must appear on its own line followed immediately by the paragraph
- End with one concrete headline recommendation that a Minister could announce at a press conference tomorrow

BEGIN POLICY BRIEF NOW:""".strip()

    else:
        prompt = f"""You are Dr. Zeno, {role_brief}. You are producing a professional economic policy brief that will be read by Ministers of Trade and Agriculture, central bank governors, IMF and World Bank country economists, and institutional commodity traders.

USER QUERY:
{query}

{f"LOCAL DATA CONTEXT:{chr(10)}{extra_context}{chr(10)}" if extra_context else ""}

No live web data was retrieved. Answer entirely from your deep expert knowledge of East African economics, agricultural commodity markets, and development policy.

{structure}

ABSOLUTE WRITING REQUIREMENTS:
- Minimum 1000 words. Maximum 1500 words. Never truncate a section.
- Include every specific number, date, institution, and policy name you know from your training data
- Write at World Bank Policy Research Paper or IMF Article IV Staff Report quality
- Name specific institutions and real programs by name
- Do NOT use bullet points, numbered lists, or markdown formatting
- Do NOT use asterisks, hashes, or any special characters
- Write every section as flowing professional paragraphs only
- Every section heading must appear on its own line followed immediately by the paragraph
- End with one concrete headline recommendation that a Minister could announce at a press conference tomorrow

BEGIN POLICY BRIEF NOW:""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
            config={
                "max_output_tokens": 8192,
                "temperature": 0.1
            }
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return "Dr. Zeno was unable to generate a response at this time. Please try again."
    except Exception as e:
        print(f"[EconomistFallback] LLM failed: {e}")
        return f"Unable to generate analysis at this time. Error: {type(e).__name__}"