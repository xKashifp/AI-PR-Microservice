"""
Generates 1500 weakly labeled examples across 5 topics
using keyword heuristics + templates.
"""
import random

TEMPLATES = {
    "product": [
        "We are launching {product} with new features for {audience}",
        "{company} releases {product} update with improved {feature}",
        "Introducing {product}: the next generation of {category}",
        "{company} unveils {product} with breakthrough {feature} capabilities",
        "New {product} version delivers enhanced {feature} for {audience}",
        "{company} announces general availability of {product} platform",
    ],
    "funding": [
        "{company} raises ${amount}M in Series {round} led by {vc}",
        "{company} secures {amount} million to expand {area}",
        "{vc} leads {amount}M investment in {company}",
        "{company} closes ${amount}M funding round to accelerate growth",
        "Investors pour ${amount}M into {company} at record valuation",
        "{company} completes Series {round} raising {amount} million from {vc}",
    ],
    "partnership": [
        "{company} partners with {partner} to deliver {service}",
        "{company} and {partner} announce strategic collaboration",
        "New integration: {company} joins forces with {partner}",
        "{company} signs multi-year agreement with {partner}",
        "{partner} selects {company} as preferred {service} provider",
        "Strategic alliance: {company} and {partner} combine platforms",
    ],
    "thought-leadership": [
        "Why {topic} will define the future of {industry}",
        "{person} on the state of {industry} in {year}",
        "Opinion: {company} CEO explains the future of {topic}",
        "{person} argues {topic} is the most important trend in {industry}",
        "Commentary: what {topic} means for {industry} professionals",
        "{company} publishes whitepaper on {topic} disrupting {industry}",
    ],
    "crisis": [
        "{company} faces backlash over {issue}",
        "Users report {issue} affecting {company} platform",
        "{company} under investigation for {issue}",
        "{company} issues apology after {issue} scandal",
        "Regulators probe {company} following {issue} allegations",
        "{company} stock falls after {issue} disclosure",
    ],
}

FILL_VARIANTS = {
    "product": ["DataLens", "CloudSync", "MetricFlow", "InsightHub", "PulseAI"],
    "company": ["Acme Corp", "TechNova", "DataSphere", "NexGen", "Vertex AI"],
    "amount": ["50", "100", "200", "500", "25", "75", "300"],
    "round": ["A", "B", "C", "D", "Seed"],
    "vc": ["Sequoia", "a16z", "Tiger Global", "Bessemer", "GV"],
    "partner": ["TechCo", "Microsoft", "Salesforce", "Oracle", "AWS"],
    "service": ["analytics", "cloud infrastructure", "AI services", "data pipelines"],
    "topic": ["AI", "Web3", "automation", "data privacy", "decentralization"],
    "industry": ["Web3", "fintech", "healthcare", "enterprise SaaS", "media"],
    "person": ["CEO Jane", "CTO Mike", "Founder Sarah", "Analyst Kim", "Expert Tom"],
    "year": ["2024", "2025", "2026"],
    "issue": ["data breach", "regulatory violations", "outage", "security flaw", "privacy leak"],
    "audience": ["enterprises", "developers", "SMBs", "data teams", "marketers"],
    "feature": ["speed", "accuracy", "scalability", "security", "reliability"],
    "category": ["SaaS", "AI platform", "developer tool", "analytics suite"],
    "area": ["APAC", "EMEA", "North America", "Latin America", "global markets"],
}

NOISE = [
    "Read more at our blog.",
    "The announcement was made today.",
    "Details to follow.",
    "This impacts the DeFi space.",
    "This marks a significant milestone.",
    "The deal is expected to close next quarter.",
    "Industry analysts called it a game changer.",
]

def generate_dataset(n: int = 1500):
    texts, labels = [], []
    topics = list(TEMPLATES.keys())
    per_topic = n // len(topics)

    for topic in topics:
        for _ in range(per_topic):
            tmpl = random.choice(TEMPLATES[topic])
            fills = {k: random.choice(v) for k, v in FILL_VARIANTS.items()}
            text = tmpl.format(**fills)
            text += " " + random.choice(NOISE)
            texts.append(text)
            labels.append(topic)

    return texts, labels
