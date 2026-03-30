# =============================================================================
# app.py  ─  EcoSort AI  |  Smart Waste Management System
# Run:  streamlit run app.py
# Pages: Dashboard | Waste Detection | Guidelines | Eco Stories
# =============================================================================

import os, sys, json, datetime, random
import numpy as np
import streamlit as st
from PIL import Image
from auth_module import render_auth_ui   # ← full auth module

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
st.markdown("""
<style>

/* Default tab text */
.stTabs [role="tab"] {
    color: #E6F4EA !important;   /* normal tabs */
    font-weight: 500;
}

/* Active tab (selected one) */
.stTabs [aria-selected="true"] {
    color: #FF4B4B !important;   /* 🔥 red like your "Paper" */
    border-bottom: 2px solid #FF4B4B !important;
}

/* Hover effect (optional nice UX) */
.stTabs [role="tab"]:hover {
    color: #4ADE80 !important;
}

</style>
""", unsafe_allow_html=True)
MODEL_PATH           = "waste_classifier_mobilenet.h5"
CONFIDENCE_THRESHOLD = 40
st.markdown("""
<style>

/* 🎯 ONLY expander headings (Plastic → Blue Bin etc.) */
.stExpander summary {
    color: #E6F4EA !important;   /* change this */
    font-weight: 600 !important;
}

/* Optional: hover */
.stExpander summary:hover {
    color: #4ADE80 !important;
}

</style>
""", unsafe_allow_html=True)
# ── Full waste database ───────────────────────────────────────────────────────
WASTE_INFO = {
    "plastic": {
        "label": "Plastic", "icon": "🧴",
        "bin": "Blue Bin", "bin_icon": "🔵",
        "bin_color": "#1A73E8", "card_bg": "#E8F0FE", "text_color": "#1A237E",
        "dot_color": "#1A73E8",
        "seg_group": "plastic",
        "category": "Dry Recyclable",
        "description": "This plastic bottle takes 450 years to decompose in landfill. When recycled, it can become part of a new pair of sneakers, a fleece jacket, or even a park bench!",
        "examples": "Bottles, bags, containers, straws, food wrapping",
        "segregation": "Sorted by NIR/IR technology by plastic type (PET, HDPE, PP, PS)",
        "recyclable": True,
        "lifecycle": [
            ("🏭", "Made from petroleum", "Extracted from crude oil — a non-renewable resource"),
            ("🛒", "Used by consumer", "Average use time: just 12 minutes for a plastic bag"),
            ("♻️", "Collected & sorted", "NIR technology identifies polymer type automatically"),
            ("🔥", "Shredded & melted", "Broken into pellets and melted at 260°C"),
            ("👟", "Reborn as new product", "Could become sneakers, fleece jacket or park bench"),
        ],
        "decompose_years": 450,
        "fun_fact": "Recycling one plastic bottle saves enough energy to power a 60W light bulb for 6 hours.",
        "eco_tip": "Rinsing plastic containers before recycling prevents contamination of the whole batch.",
        "upcycle": [
            {"title": "Bottle Bird Feeder", "emoji": "🐦",
             "steps": "Cut a hole in the side, insert a wooden spoon as a perch, fill with birdseed. Hang outdoors!",
             "difficulty": "Easy", "time": "15 mins"},
            {"title": "Plastic Bottle Planter", "emoji": "🌱",
             "steps": "Cut the top off, poke drainage holes in the bottom, fill with soil and plant herbs.",
             "difficulty": "Easy", "time": "10 mins"},
            {"title": "DIY Piggy Bank", "emoji": "🐷",
             "steps": "Seal the cap, cut a coin slot on top, paint it — a creative savings jar for kids!",
             "difficulty": "Easy", "time": "20 mins"},
        ],
        "actions": ["Rinse before recycling","Remove caps separately","Flatten to save space","Sort by plastic type"],
        "do_not": ["No dirty/greasy plastic","No plastic bags in recycling","No mixing with wet waste"],
    },
    "paper": {
        "label": "Paper", "icon": "📄",
        "bin": "Blue Bin", "bin_icon": "🔵",
        "bin_color": "#1A73E8", "card_bg": "#E8F0FE", "text_color": "#1A237E",
        "dot_color": "#1A73E8",
        "seg_group": "plastic",
        "category": "Dry Recyclable",
        "description": "Paper decomposes in 2–6 weeks when composted, but recycling is far better — it saves trees and uses 40% less energy than making new paper.",
        "examples": "Newspapers, magazines, cardboard, office paper, envelopes",
        "segregation": "Must be kept dry and clean — wet paper cannot be recycled",
        "recyclable": True,
        "lifecycle": [
            ("🌳", "Trees harvested", "1 ton of paper requires 24 trees to produce"),
            ("🏭", "Pulped & processed", "Wood chips are boiled with chemicals to create pulp"),
            ("📰", "Made into paper", "Pulp is spread, dried and rolled into sheets"),
            ("🗑️", "Used & discarded", "Average office worker uses 10,000 sheets per year"),
            ("♻️", "Recycled into new paper", "Can be recycled 5–7 times before fibres shorten too much"),
        ],
        "decompose_years": 0.1,
        "fun_fact": "Recycling 1 ton of paper saves 17 trees, 7,000 gallons of water and 4,000 kWh of electricity.",
        "eco_tip": "Keeping paper dry and clean ensures it can be fully recycled into new paper products.",
        "upcycle": [
            {"title": "Handmade Notebook", "emoji": "📓",
             "steps": "Fold used paper in half, stack pages, sew the spine with thread, add a cardboard cover.",
             "difficulty": "Medium", "time": "45 mins"},
            {"title": "Paper Mâché Bowl", "emoji": "🥣",
             "steps": "Mix flour and water paste, layer torn paper strips over a balloon, let dry, pop balloon.",
             "difficulty": "Medium", "time": "2 days (drying)"},
            {"title": "Seed Paper", "emoji": "🌸",
             "steps": "Blend paper with water, mix in wildflower seeds, press flat and dry — plant it later!",
             "difficulty": "Easy", "time": "1 day"},
        ],
        "actions": ["Keep dry and clean","Remove staples and clips","Flatten cardboard boxes","Shred sensitive documents"],
        "do_not": ["No wet or oily paper","No tissue or paper towels","No mixing with food waste"],
    },
    "metal": {
        "label": "Metal", "icon": "🥫",
        "bin": "Blue Bin", "bin_icon": "🔵",
        "bin_color": "#1A73E8", "card_bg": "#E8F0FE", "text_color": "#1A237E",
        "dot_color": "#1A73E8",
        "seg_group": "metal",
        "category": "Dry Recyclable",
        "description": "Aluminium cans can be recycled and back on the shelf in just 60 days. Recycling aluminium uses 95% less energy than making it from raw ore!",
        "examples": "Aluminium cans, foil, steel containers, iron, bottle caps",
        "segregation": "Magnetic separation (ferrous/iron/steel) or eddy currents (aluminium)",
        "recyclable": True,
        "lifecycle": [
            ("⛏️", "Bauxite mined", "Aluminium ore is strip-mined — very energy intensive"),
            ("🔥", "Smelted at 960°C", "Enormous amounts of electricity needed for smelting"),
            ("🥤", "Formed into products", "Rolled into cans, foil or containers"),
            ("🗑️", "Disposed after use", "Average can is used for just 3 hours"),
            ("⚡", "Recycled in 60 days", "Back on the shelf using only 5% of original energy"),
        ],
        "decompose_years": 200,
        "fun_fact": "The aluminium in a recycled can could be back on the shelf as a new can within 60 days.",
        "eco_tip": "Aluminium can be recycled forever — every can you recycle matters!",
        "upcycle": [
            {"title": "Tin Can Lantern", "emoji": "🕯️",
             "steps": "Fill can with water, freeze solid, hammer nail holes in a pattern, add tea light inside.",
             "difficulty": "Medium", "time": "1 day (freezing)"},
            {"title": "Herb Garden Pots", "emoji": "🌿",
             "steps": "Remove label, punch drainage holes in base, paint with chalkboard paint, plant herbs.",
             "difficulty": "Easy", "time": "20 mins"},
            {"title": "Desk Organiser", "emoji": "✏️",
             "steps": "Collect several cans, spray paint same colour, glue together in a cluster — instant organiser!",
             "difficulty": "Easy", "time": "30 mins"},
        ],
        "actions": ["Rinse food cans thoroughly","Crush cans to save space","Scrunch foil into a ball","Remove paper labels"],
        "do_not": ["No non-empty aerosol cans","No paint cans with dried paint","No sharp metal loose in bin"],
    },
    "glass": {
        "label": "Glass", "icon": "🍾",
        "bin": "Grey Bin", "bin_icon": "⚫",
        "bin_color": "#5F6368", "card_bg": "#F1F3F4", "text_color": "#202124",
        "dot_color": "#5F6368",
        "seg_group": "metal",
        "category": "Dry Recyclable",
        "description": "Glass can be recycled endlessly without any loss in quality or purity. A glass jar has infinite lives — every bottle you recycle could become another bottle forever.",
        "examples": "Glass bottles, jars, food and beverage containers",
        "segregation": "Sorted by colour — clear (flint), green, amber/brown",
        "recyclable": True,
        "lifecycle": [
            ("🏖️", "Sand mined", "Glass is made from silica sand, soda ash and limestone"),
            ("🔥", "Melted at 1700°C", "Enormous energy needed to melt raw materials"),
            ("🍾", "Formed into bottles", "Blown or pressed into shape while molten"),
            ("🛒", "Used by consumer", "Average glass jar used 1–2 times before disposal"),
            ("♾️", "Recycled forever", "Crushed into cullet, remelted at lower temp, reborn!"),
        ],
        "decompose_years": 1000000,
        "fun_fact": "Glass takes over 1 million years to fully decompose — but it can be recycled forever at lower cost.",
        "eco_tip": "Every recycled glass bottle saves enough energy to power a TV for 1.5 hours.",
        "upcycle": [
            {"title": "Mini Terrarium", "emoji": "🌵",
             "steps": "Clean the jar, add pebbles for drainage, charcoal layer, potting soil, then small succulents.",
             "difficulty": "Easy", "time": "30 mins"},
            {"title": "Spice Container", "emoji": "🧂",
             "steps": "Clean thoroughly, add a chalkboard label, fill with your loose spices — zero-waste kitchen!",
             "difficulty": "Easy", "time": "5 mins"},
            {"title": "Fairy Light Jar", "emoji": "✨",
             "steps": "Coil LED fairy lights inside the jar, leave wire over the rim for battery pack. Magical!",
             "difficulty": "Easy", "time": "10 mins"},
        ],
        "actions": ["Rinse bottles and jars","Remove metal lids separately","Sort by colour if required","Take to bottle bank if needed"],
        "do_not": ["No broken glass in recycling bin","No ceramics or mirrors","No drinking glasses or Pyrex"],
    },
    "organic": {
        "label": "Organic", "icon": "🌿",
        "bin": "Green Bin", "bin_icon": "🟢",
        "bin_color": "#1E8E3E", "card_bg": "#E6F4EA", "text_color": "#1B5E20",
        "dot_color": "#1E8E3E",
        "seg_group": "organic",
        "category": "Wet / Biodegradable",
        "description": "Organic waste decomposes in 2–4 weeks when composted properly. Instead of producing methane in landfill, it can become rich fertiliser that feeds new plants!",
        "examples": "Food scraps, fruit/vegetable peels, garden waste, tea bags, coffee grounds",
        "segregation": "Composted or bio-methanation to create fertiliser or biogas",
        "recyclable": False,
        "lifecycle": [
            ("🌱", "Grown from soil", "Food is grown using soil nutrients, water and sunlight"),
            ("🍎", "Consumed", "Food provides energy and nutrition"),
            ("🗑️", "Scraps discarded", "Food scraps in landfill produce methane — 25x worse than CO2"),
            ("🪱", "Composted", "Microorganisms and worms break it down in weeks"),
            ("🌻", "Returns to soil", "Rich compost feeds new plants — perfect circular cycle"),
        ],
        "decompose_years": 0.05,
        "fun_fact": "Food waste in landfill produces methane — a greenhouse gas 25x more potent than CO2. Composting stops this!",
        "eco_tip": "Composting food waste at home reduces your carbon footprint and produces free fertiliser for your garden.",
        "upcycle": [
            {"title": "Home Compost Bin", "emoji": "🪣",
             "steps": "Use a lidded bin or plastic crate, add alternating layers of green (food) and brown (leaves), keep moist.",
             "difficulty": "Easy", "time": "Ongoing"},
            {"title": "Citrus Peel Cleaner", "emoji": "🍋",
             "steps": "Pack citrus peels into a jar, cover with white vinegar, wait 2 weeks, strain. Natural cleaner!",
             "difficulty": "Easy", "time": "2 weeks"},
            {"title": "Vegetable Scrap Broth", "emoji": "🍲",
             "steps": "Collect onion skins, carrot tops, celery ends in a freezer bag. When full, boil into stock.",
             "difficulty": "Easy", "time": "45 mins"},
        ],
        "actions": ["Compost food scraps and peels","Include garden waste — grass and leaves","Use sealed bin for odour control","Layer dry and wet material in compost"],
        "do_not": ["No meat or dairy in home compost","No mixing with dry recyclables","No plastic bags — even biodegradable ones"],
    },
    "general": {
        "label": "General Waste", "icon": "🗑️",
        "bin": "Red Bin", "bin_icon": "🔴",
        "bin_color": "#D93025", "card_bg": "#FCE8E6", "text_color": "#B31412",
        "dot_color": "#D93025",
        "seg_group": "general",
        "category": "Residual / Non-Recyclable",
        "description": "This is mixed and contaminated waste that ends up in landfill or waste-to-energy plants. The best action is to reduce this type of waste at source by choosing recyclable packaging.",
        "examples": "Dirty wrappers, hygiene products, thermocol, broken items",
        "segregation": "Sent to landfill or waste-to-energy plants for processing",
        "recyclable": False,
        "lifecycle": [
            ("🏭", "Manufactured", "Made from mixed materials — hard to separate later"),
            ("🛒", "Purchased & used", "Single-use items used once then discarded"),
            ("🗑️", "Sent to landfill", "Buried underground — takes decades to break down"),
            ("☠️", "Leaches toxins", "Chemicals seep into groundwater over decades"),
            ("⚡", "Or waste-to-energy", "Some is burned to generate electricity — last resort"),
        ],
        "decompose_years": 500,
        "fun_fact": "The average person generates 4.5 lbs of trash per day. 75% of it could be recycled or composted.",
        "eco_tip": "Reducing general waste starts with buying less — choose products with recyclable or compostable packaging.",
        "upcycle": [
            {"title": "Reduce First!", "emoji": "🛑",
             "steps": "Before buying anything, ask: Do I really need this? Can I borrow it, buy second-hand, or find a recyclable alternative?",
             "difficulty": "Mindset shift", "time": "Always"},
            {"title": "Repair Don't Replace", "emoji": "🔧",
             "steps": "Broken item? Look up a repair tutorial on YouTube. Many things — clothes, electronics, furniture — can be fixed.",
             "difficulty": "Varies", "time": "Varies"},
        ],
        "actions": ["Remove all recyclables first","Wrap sharp items securely","Choose recyclable packaging next time","Check local guidelines"],
        "do_not": ["No hazardous waste (batteries, chemicals)","No e-waste or electronics","No medical or clinical waste"],
    },
}

# ── Session state ─────────────────────────────────────────────────────────────
if "history"     not in st.session_state: st.session_state.history     = []
if "page"        not in st.session_state: st.session_state.page        = "Dashboard"
if "scan_mode"   not in st.session_state: st.session_state.scan_mode   = "upload"
if "get_started" not in st.session_state: st.session_state.get_started = False
if "user_info"   not in st.session_state: st.session_state.user_info   = None
if "auth_tab"    not in st.session_state: st.session_state.auth_tab    = "login"

# ── Page config ───────────────────────────────────────────────────────────────
# ── Sidebar state ──
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ── Page config (ONLY ONCE, MUST BE AT TOP) ──
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded" 
)



# ── Styles + mobile sidebar toggle ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp, .main, section[data-testid="stMain"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0A1628 0%, #0D2B1A 50%, #051A10 100%) !important;
    min-height: 100vh !important;
}
.main .block-container { background: transparent !important; padding-top: 2rem !important; }
#MainMenu, footer { visibility: hidden !important; }
header { visibility: visible !important; }


/* ── Sidebar colours ── */
[data-testid="stSidebar"], section[data-testid="stSidebar"] {
    background: #0A1420 !important;
    border-right: 1px solid rgba(74,222,128,0.18) !important;
    min-height: 100vh !important;
}
[data-testid="stSidebar"] > div:first-child { background: #0A1420 !important; padding: 20px 16px !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; border: none !important;
    text-align: left !important; padding: 10px 14px !important;
    border-radius: 10px !important; font-size: 14px !important;
    font-weight: 500 !important; color: rgba(255,255,255,0.8) !important;
    width: 100% !important; margin-bottom: 2px !important; transition: all 0.18s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(74,222,128,0.12) !important; color: #4ADE80 !important;
}

/* ───────── TARGETED TEXT COLOR FIX ───────── */

/* ONLY your custom components */
.panel,
.kpi,
.result-card,
.seg-panel,
.eco-card,
.up-card,
.lc-step {
    color: #FFFFFF !important;
}

/* Fix text inside these components */
.panel *,
.kpi *,
.result-card *,
.seg-panel *,
.eco-card *,
.up-card *,
.lc-step * {
    color: inherit !important;
}

/* Sidebar user card (your profile box) */
section[data-testid="stSidebar"] .panel,
section[data-testid="stSidebar"] .panel * {
    color: #FFFFFF !important;
}

/* Fix faded text (like email, labels) */
small, span, p {
    opacity: 1 !important;
}
.status-bar{font-size:11px;font-weight:600;color:#4ADE80!important;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px}
.page-title{font-size:30px;font-weight:900;color:#FFFFFF!important;margin:0 0 24px;letter-spacing:-0.5px}
.kpi{background:rgba(255,255,255,0.07);border-radius:16px;padding:22px;border:1px solid rgba(74,222,128,0.15);position:relative;backdrop-filter:blur(8px)}
.kpi-badge{font-size:11px;font-weight:700;padding:3px 8px;border-radius:20px;position:absolute;top:14px;right:14px}
.kpi-badge.g{background:rgba(30,142,62,0.25);color:#4ADE80!important}
.kpi-badge.r{background:rgba(217,48,37,0.2);color:#F87171!important}
.kpi-lbl{font-size:13px;color:rgba(255,255,255,0.5)!important;margin:10px 0 3px;font-weight:500}
.kpi-val{font-size:34px;font-weight:900;color:#FFFFFF!important;margin:0}
.panel{background:rgba(255,255,255,0.07);border-radius:16px;padding:24px;border:1px solid rgba(74,222,128,0.12);backdrop-filter:blur(8px)}
.panel-title{font-size:15px;font-weight:700;color:#FFFFFF!important;margin:0 0 16px}
.bin-box{background:rgba(30,142,62,0.15);border-radius:12px;padding:14px 16px;margin-top:20px;border:1px solid rgba(74,222,128,0.25)}
.bin-lbl{font-size:10px;font-weight:700;letter-spacing:.8px;color:#4ADE80!important;text-transform:uppercase;margin-bottom:8px}
.result-card{background:rgba(255,255,255,0.08);border-radius:20px;padding:36px 28px;text-align:center;border:1px solid rgba(74,222,128,0.18);box-shadow:0 8px 32px rgba(0,0,0,.3);backdrop-filter:blur(12px)}
.r-icon{width:76px;height:76px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 14px;font-size:34px}
.r-name{font-size:26px;font-weight:900;color:#FFFFFF!important;margin:0 0 8px}
.r-desc{font-size:13px;color:rgba(255,255,255,0.6)!important;line-height:1.6;margin:0 auto 16px;max-width:280px}
.r-conf-lbl{font-size:10px;font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.5)!important;text-transform:uppercase;margin-bottom:3px}
.r-conf-val{font-size:22px;font-weight:900;color:#4ADE80!important}
.seg-panel{background:rgba(255,255,255,0.07);border-radius:20px;padding:22px;border:1px solid rgba(74,222,128,0.12);backdrop-filter:blur(8px)}
.seg-title{font-size:16px;font-weight:700;color:#FFFFFF!important;margin:0 0 16px}
.seg-row{display:flex;align-items:flex-start;padding:12px 14px;border-radius:12px;margin-bottom:8px;background:rgba(255,255,255,0.05);border:1.5px solid transparent}
.seg-row.active{border-color:#4ADE80!important;background:rgba(74,222,128,0.08)!important}
.seg-dot{width:11px;height:11px;border-radius:50%;margin-top:4px;flex-shrink:0}
.seg-ml{margin-left:11px}
.seg-t{font-size:14px;font-weight:600;color:#FFFFFF!important;margin:0 0 2px}
.seg-s{font-size:12px;color:rgba(255,255,255,0.5)!important;margin:0}
.eco-card{background:linear-gradient(135deg,#1E8E3E,#16A34A);border-radius:16px;padding:18px 22px;margin-top:14px;box-shadow:0 4px 20px rgba(30,142,62,0.4)}
.eco-t{font-size:15px;font-weight:700;color:#FFF!important;margin:0 0 6px}
.eco-p{font-size:13px;color:#D5F0DC!important;line-height:1.6;margin:0}
.lc-step{display:flex;align-items:flex-start;gap:14px;padding:14px 16px;border-radius:12px;background:rgba(255,255,255,0.05);margin-bottom:10px;border-left:4px solid}
.lc-icon{font-size:28px;flex-shrink:0}
.lc-title{font-size:14px;font-weight:700;color:#FFFFFF!important;margin:0 0 3px}
.lc-desc{font-size:12px;color:rgba(255,255,255,0.55)!important;margin:0}
.up-card{background:rgba(255,255,255,0.07);border-radius:16px;padding:20px;border:1px solid rgba(74,222,128,0.12);margin-bottom:12px;backdrop-filter:blur(8px)}
.up-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.up-emoji{font-size:28px}
.up-title{font-size:15px;font-weight:700;color:#FFFFFF!important;margin:0}
.up-meta{font-size:11px;color:rgba(255,255,255,0.5)!important;margin:2px 0 0}
.up-steps{font-size:13px;color:rgba(255,255,255,0.7)!important;line-height:1.6;margin:0}
.do-item{background:rgba(30,142,62,0.15);border-radius:10px;padding:10px 14px;margin:5px 0;border-left:4px solid #4ADE80}
.do-item span{font-size:13px;font-weight:500;color:#86EFAC!important}
.dont-item{background:rgba(217,48,37,0.12);border-radius:10px;padding:10px 14px;margin:5px 0;border-left:4px solid #F87171}
.dont-item span{font-size:13px;font-weight:500;color:#FCA5A5!important}
.h-row{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-radius:10px;background:rgba(255,255,255,0.05);margin-bottom:6px;border:1px solid rgba(255,255,255,0.08)}
.h-w{font-size:13px;font-weight:600;color:#FFFFFF!important}
.h-t{font-size:11px;color:rgba(255,255,255,0.4)!important}
.h-c{font-size:13px;font-weight:700;color:#4ADE80!important}
.stTextInput input,.stTextArea textarea{background:rgba(255,255,255,0.07)!important;border-radius:10px!important;border:1px solid rgba(74,222,128,0.2)!important;color:#FFFFFF!important;font-size:14px!important}
.stButton>button{border-radius:12px!important;font-weight:600!important;font-size:14px!important}
.cam-result-box{background:rgba(255,255,255,0.07);border-radius:16px;padding:20px;border:2px solid #4ADE80;margin-top:16px}
.gs-hero{min-height:100vh;background:linear-gradient(135deg,#0A1628 0%,#0D2B1A 50%,#051A10 100%);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 24px;position:relative;overflow:hidden;text-align:center}
.gs-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(74,222,128,0.12);border:1px solid rgba(74,222,128,0.3);border-radius:100px;padding:8px 18px;font-size:13px;font-weight:600;color:#4ADE80;margin-bottom:32px}
.gs-headline{font-size:clamp(40px,7vw,72px);font-weight:900;color:#FFFFFF;line-height:1.08;margin-bottom:20px;letter-spacing:-1.5px}
.gs-headline-green{color:#4ADE80}
.gs-sub{font-size:18px;color:rgba(255,255,255,0.55);max-width:580px;line-height:1.7;margin-bottom:52px}
.gs-stats{display:flex;gap:48px;justify-content:center;flex-wrap:wrap;margin-bottom:72px}
.gs-stat-val{font-size:32px;font-weight:900;color:#4ADE80;margin-bottom:4px}
.gs-stat-lbl{font-size:13px;color:rgba(255,255,255,0.5);font-weight:500}
.gs-features{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px;max-width:960px;width:100%;margin-bottom:64px}
.gs-feat-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:20px;padding:28px 24px;text-align:left;backdrop-filter:blur(12px);transition:all 0.25s}
.gs-feat-card:hover{background:rgba(255,255,255,0.09);border-color:rgba(74,222,128,0.3);transform:translateY(-3px)}
.gs-feat-icon{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:16px}
.gs-feat-title{font-size:16px;font-weight:700;color:#FFFFFF;margin-bottom:8px}
.gs-feat-desc{font-size:13px;color:rgba(255,255,255,0.55);line-height:1.6}
.gs-how-title{font-size:28px;font-weight:800;color:#FFFFFF;margin-bottom:40px}
.gs-steps{display:flex;justify-content:center;max-width:820px;width:100%;flex-wrap:wrap;margin-bottom:72px}
.gs-step{flex:1;min-width:160px;text-align:center;padding:0 16px;position:relative}
.gs-step:not(:last-child)::after{content:"→";position:absolute;right:-10px;top:24px;color:rgba(74,222,128,0.4);font-size:22px}
.gs-step-num{width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#1E8E3E22,#1E8E3E44);border:2px solid #4ADE8066;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:900;color:#4ADE80;margin:0 auto 14px}
.gs-step-title{font-size:14px;font-weight:700;color:#FFFFFF;margin-bottom:6px}
.gs-step-desc{font-size:12px;color:rgba(255,255,255,0.45);line-height:1.5}
.gs-footer-bar{border-top:1px solid rgba(255,255,255,0.08);padding-top:28px;width:100%;max-width:960px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.gs-footer-logo{font-size:16px;font-weight:800;color:#4ADE80}
.gs-footer-copy{font-size:12px;color:rgba(255,255,255,0.3)}
.gs-orb1{position:absolute;width:400px;height:400px;border-radius:50%;background:radial-gradient(circle,rgba(30,142,62,0.18) 0%,transparent 70%);top:-100px;left:-100px;pointer-events:none;animation:gs-float1 8s ease-in-out infinite}
.gs-orb2{position:absolute;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(26,115,232,0.12) 0%,transparent 70%);bottom:-80px;right:-80px;pointer-events:none;animation:gs-float2 10s ease-in-out infinite}
.gs-orb3{position:absolute;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(74,222,128,0.1) 0%,transparent 70%);top:40%;right:10%;pointer-events:none;animation:gs-float1 6s ease-in-out infinite reverse}
@keyframes gs-float1{0%,100%{transform:translate(0,0)}50%{transform:translate(30px,-30px)}}
@keyframes gs-float2{0%,100%{transform:translate(0,0)}50%{transform:translate(-25px,25px)}}
.gs-cta-wrap .stButton>button{background:linear-gradient(135deg,#1E8E3E,#16A34A)!important;color:#FFFFFF!important;border:none!important;border-radius:14px!important;font-size:17px!important;font-weight:700!important;box-shadow:0 8px 32px rgba(30,142,62,0.5)!important;height:58px!important;width:100%!important}
.gs-cta-wrap .stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 12px 40px rgba(30,142,62,0.7)!important}
</style>
""", unsafe_allow_html=True)

# ── Mobile sidebar toggle ─────────────────────────────────────────────────────
# Uses st.components.v1.html which creates its own iframe.
# That iframe IS same-origin with Streamlit's shell, so window.parent works.

# ── Load model (cached for speed) ────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        return None

@st.cache_data
def load_classes():
    if not os.path.exists("class_indices.json"): return {}
    with open("class_indices.json") as f: idx = json.load(f)
    return {v: k for k, v in idx.items()}

model       = load_model()
class_names = load_classes() if model else {}

import base64, io, urllib.request, urllib.error

def _pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def _claude_classify(pil_img):
    b64 = _pil_to_b64(pil_img)
    categories = list(WASTE_INFO.keys())
    prompt = (
        "You are a waste classification expert. Look at this image and identify what waste type it is.\n"
        f"Choose EXACTLY ONE category from: {categories}.\n"
        "Also give a confidence percentage (0-100).\n"
        "Respond ONLY with valid JSON like: {\"waste\": \"plastic\", \"confidence\": 92}\n"
        "No explanation, no markdown, just the JSON."
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
        result = json.loads(text)
        waste = result.get("waste","general").lower()
        if waste not in WASTE_INFO: waste = "general"
        conf  = float(result.get("confidence", 85))
        scores = {k: round(2.0,1) for k in WASTE_INFO}
        scores[waste] = round(conf, 1)
        return waste, round(conf,1), scores
    except Exception:
        return "general", 60.0, {k:10.0 for k in WASTE_INFO}

def predict_image(pil_img):
    if model is not None:
        x = np.array(pil_img.resize((224,224)), dtype="float32")
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        preds   = model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        conf    = float(preds[top_idx]) * 100
        waste   = class_names[top_idx]
        if conf < CONFIDENCE_THRESHOLD: waste = "general"
        scores  = {class_names[i]: float(round(preds[i]*100,1)) for i in range(len(preds))}
        return waste, round(conf,1), scores
    else:
        return _claude_classify(pil_img)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — GET STARTED (landing)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.get_started:
    st.markdown("<style>.main .block-container{padding:0!important;max-width:100%!important}</style>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="gs-hero">
      <div class="gs-orb1"></div><div class="gs-orb2"></div><div class="gs-orb3"></div>
      <div class="gs-badge"><span>🌿</span> AI-Powered Smart Waste Management</div>
      <div class="gs-headline">Sort Smarter.<br><span class="gs-headline-green">Live Greener.</span></div>
      <div class="gs-sub">EcoSort AI uses computer vision to identify waste in seconds, tell you exactly which bin to use, and show you how to upcycle it into something beautiful.</div>
      <div class="gs-stats">
        <div><div class="gs-stat-val">6</div><div class="gs-stat-lbl">Waste Categories</div></div>
        <div><div class="gs-stat-val">95%</div><div class="gs-stat-lbl">Detection Accuracy</div></div>
        <div><div class="gs-stat-val">60%</div><div class="gs-stat-lbl">Landfill Reduction</div></div>
        <div><div class="gs-stat-val">&#8734;</div><div class="gs-stat-lbl">Upcycle Ideas</div></div>
      </div>
      <div class="gs-features">
        <div class="gs-feat-card"><div class="gs-feat-icon" style="background:rgba(26,115,232,0.15)">📷</div><div class="gs-feat-title">Instant AI Detection</div><div class="gs-feat-desc">Upload a photo or use your live camera — get a classification in under 2 seconds.</div></div>
        <div class="gs-feat-card"><div class="gs-feat-icon" style="background:rgba(30,142,62,0.15)">🗂️</div><div class="gs-feat-title">Smart Segregation</div><div class="gs-feat-desc">Know exactly which bin — Green, Blue, Grey or Red — with a visual AR guide.</div></div>
        <div class="gs-feat-card"><div class="gs-feat-icon" style="background:rgba(251,188,4,0.15)">💡</div><div class="gs-feat-title">Upcycling Ideas</div><div class="gs-feat-desc">Turn your waste into art, planters, and more. Step-by-step DIY projects for every item.</div></div>
        <div class="gs-feat-card"><div class="gs-feat-icon" style="background:rgba(234,67,53,0.15)">🌍</div><div class="gs-feat-title">Eco Stories</div><div class="gs-feat-desc">See the full lifecycle of every waste type and understand your real environmental impact.</div></div>
      </div>
      <div class="gs-how-title">How it works</div>
      <div class="gs-steps">
        <div class="gs-step"><div class="gs-step-num">1</div><div class="gs-step-title">Snap or Upload</div><div class="gs-step-desc">Take a photo of any waste item with your camera or upload from gallery</div></div>
        <div class="gs-step"><div class="gs-step-num">2</div><div class="gs-step-title">AI Classifies</div><div class="gs-step-desc">MobileNetV2 model identifies the waste type with confidence score</div></div>
        <div class="gs-step"><div class="gs-step-num">3</div><div class="gs-step-title">Get Guidance</div><div class="gs-step-desc">See which bin, how to prepare it, and eco tips tailored to the item</div></div>
        <div class="gs-step"><div class="gs-step-num">4</div><div class="gs-step-title">Upcycle It</div><div class="gs-step-desc">Browse creative DIY ideas to repurpose instead of throwing away</div></div>
      </div>
      <div class="gs-footer-bar">
        <div class="gs-footer-logo">♻️ EcoSort AI</div>
        <div class="gs-footer-copy">Built with TensorFlow · MobileNetV2 · Streamlit</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<style>.main .block-container{padding-top:0!important;padding-bottom:0!important}</style>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀  Get Started", use_container_width=True, type="primary"):
            st.session_state.get_started = True
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN  — Login / Register / Google Sign-In
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.user_info is None:
    render_auth_ui()
    st.stop()
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 0 24px">
      <div style="background:linear-gradient(135deg,#1E8E3E,#16A34A);border-radius:12px;
                  width:44px;height:44px;display:flex;align-items:center;justify-content:center;
                  font-size:22px;box-shadow:0 4px 12px rgba(30,142,62,0.4)">♻️</div>
      <div>
        <div style="font-size:18px;font-weight:800;color:#FFFFFF">EcoSort AI</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.4)">Smart Waste Management</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.user_info:
        u = st.session_state.user_info
        provider_badge = "🔵 Google" if u.get("provider") == "google" else "📧 Email"
        st.markdown(f"""
        <div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);
                    border-radius:14px;padding:12px 14px;margin-bottom:20px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <img src="{u['picture']}" style="width:36px;height:36px;border-radius:50%;border:2px solid #4ADE80;flex-shrink:0"/>
            <div>
              <div style="font-size:13px;font-weight:700;color:#FFFFFF">{u['name']}</div>
              <div style="font-size:11px;color:rgba(255,255,255,0.4)">{u['email']}</div>
            </div>
          </div>
          <div style="font-size:10px;color:rgba(74,222,74,0.7);font-weight:600">{provider_badge}</div>
        </div>""", unsafe_allow_html=True)

    PAGES = [("🏠","Dashboard"),("📷","Waste Detection"),("📖","Guidelines"),("🌍","Eco Stories")]
    for icon, pg in PAGES:
        is_active = st.session_state.page == pg
        if is_active:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(30,142,62,0.3),rgba(74,222,128,0.15));
                        border:1px solid rgba(74,222,128,0.35);border-radius:10px;padding:12px 16px;
                        margin-bottom:4px;display:flex;align-items:center;gap:10px">
              <span style="font-size:17px">{icon}</span>
              <span style="font-size:14px;font-weight:700;color:#4ADE80">{pg}</span>
              <span style="margin-left:auto;color:#4ADE80;font-size:16px">›</span>
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {pg}", key=f"nav_{pg}", use_container_width=True):
                st.session_state.page = pg
                st.rerun()

    st.markdown("""
    <div class="bin-box">
      <div class="bin-lbl">🟢 Bin Status</div>
      <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:8px;margin:4px 0">
        <div style="background:linear-gradient(90deg,#4ADE80,#1E8E3E);border-radius:6px;height:8px;width:72%"></div>
      </div>
      <div style="font-size:12px;color:rgba(255,255,255,0.4);margin-top:6px">Next pickup in 4 hours</div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.user_info:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True):
            st.session_state.user_info   = None
            st.session_state.get_started = False
            st.session_state.auth_tab    = "login"
            for key in ["connected", "oauth_token"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
# ── Sidebar toggle button ──
col1, col2 = st.columns([1, 20])
with col1:
    if st.button("☰", use_container_width=True):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Dashboard":
    st.markdown('<div class="status-bar">System Status: Active</div>', unsafe_allow_html=True)
    user_name = st.session_state.user_info["name"] if st.session_state.user_info else "Admin"
    st.markdown(f'<div class="page-title">Welcome back, {user_name} 👋</div>', unsafe_allow_html=True)

    total = len(st.session_state.history)
    rec   = sum(1 for h in st.session_state.history if WASTE_INFO.get(h["waste"],{}).get("recyclable", False))
    rate  = int(rec/total*100) if total > 0 else 74

    k1,k2,k3 = st.columns(3)
    for col, (badge_cls, badge_txt, ico, lbl, val) in zip([k1,k2,k3],[
        ("g","+12%","📷","Total Detections", total),
        ("g","+5.4%","♻️","Recycling Rate", f"{rate}%"),
        ("r","0","🚨","Active Reports", 0),
    ]):
        col.markdown(f"""
        <div class="kpi">
          <span class="kpi-badge {badge_cls}">{badge_txt}</span>
          <div style="font-size:26px;margin-bottom:4px">{ico}</div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-val">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns([3,2], gap="medium")
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">Waste Distribution</div>', unsafe_allow_html=True)
        if st.session_state.history:
            import pandas as pd
            counts = {}
            for h in st.session_state.history:
                counts[h["waste"]] = counts.get(h["waste"],0)+1
            st.bar_chart(pd.DataFrame({"Waste":list(counts.keys()),"Count":list(counts.values())}).set_index("Waste"), height=200)
        else:
            st.markdown("""<div style="height:180px;display:flex;align-items:center;justify-content:center;color:#9AA0A6!important;font-size:14px">No detection data yet — scan some waste!</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="panel-title">⏱ Recent Activity</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for h in st.session_state.history[-5:][::-1]:
                info = WASTE_INFO.get(h["waste"], WASTE_INFO["general"])
                st.markdown(f"""
                <div class="h-row">
                  <div><div class="h-w">{info['icon']} {info['label']}</div><div class="h-t">{h['time']}</div></div>
                  <div class="h-c">{h['conf']}%</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="height:160px;display:flex;align-items:center;justify-content:center;color:#9AA0A6!important;font-size:14px">No recent detections</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — WASTE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Waste Detection":
    st.markdown('<div class="status-bar">System Status: Active</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Waste Classification</div>', unsafe_allow_html=True)

    if model is None:
        st.info("🤖 Local model not found — using **Claude AI Vision** for waste classification. Upload an image to get started!", icon="✨")

    left, right = st.columns([3,2], gap="large")

    with left:
        t1,t2 = st.columns(2)
        with t1:
            if st.button("📁  Upload Image",
                type="primary" if st.session_state.scan_mode=="upload" else "secondary",
                use_container_width=True):
                st.session_state.scan_mode = "upload"
                st.session_state.pop("result", None); st.rerun()
        with t2:
            if st.button("📷  Live Camera",
                type="primary" if st.session_state.scan_mode=="camera" else "secondary",
                use_container_width=True):
                st.session_state.scan_mode = "camera"
                st.session_state.pop("result", None); st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.scan_mode == "upload":
            if "result" not in st.session_state:
                up = st.file_uploader("", type=["jpg","png","jpeg","webp"], label_visibility="collapsed")
                if not up:
                    st.markdown("""
                    <div style="background:#FFF;border-radius:20px;border:2px dashed #DADCE0;padding:60px 30px;text-align:center">
                      <div style="font-size:44px;margin-bottom:10px">📤</div>
                      <div style="font-size:17px;font-weight:600;color:#202124!important">Drop waste image here</div>
                      <div style="font-size:13px;color:#5F6368!important;margin-top:4px">JPG · PNG · JPEG · WEBP</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    img = Image.open(up).convert("RGB")
                    st.image(img, use_column_width=True)
                    with st.spinner("🔍 Analysing with AI ..."):
                        waste, conf, scores = predict_image(img)
                    st.session_state.result = {"waste":waste,"conf":conf,"scores":scores,"img":img}
                    st.session_state.history.append({"waste":waste,"conf":conf,"time":datetime.datetime.now().strftime("%H:%M:%S")})
                    st.rerun()
            else:
                st.image(st.session_state.result["img"], use_column_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                res  = st.session_state.result
                info = WASTE_INFO[res["waste"]]
                bin_colors = {"Blue Bin":"#1A73E8","Grey Bin":"#5F6368","Green Bin":"#1E8E3E","Red Bin":"#D93025"}
                bc = bin_colors.get(info["bin"],"#1A73E8")
                st.markdown("**🥽 AR Bin Overlay — Where to place it:**")
                ar_html = f"""
                <div style="background:#111827;border-radius:16px;overflow:hidden;position:relative;height:220px;margin-bottom:8px">
                  <canvas id="arCanvas" style="width:100%;height:220px"></canvas>
                  <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:white;font-family:Inter,sans-serif;pointer-events:none">
                    <div style="font-size:12px;opacity:.7;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">Place item here →</div>
                    <div style="font-size:22px;font-weight:800">{info['bin_icon']} {info['bin']}</div>
                  </div>
                  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                  <script>
                    (function(){{
                      const canvas = document.getElementById('arCanvas');
                      const renderer = new THREE.WebGLRenderer({{canvas, antialias:true, alpha:true}});
                      renderer.setSize(canvas.offsetWidth||600, 220);
                      renderer.setClearColor(0x111827,1);
                      const scene  = new THREE.Scene();
                      const camera = new THREE.PerspectiveCamera(45, (canvas.offsetWidth||600)/220, 0.1, 100);
                      camera.position.set(0, 1.6, 4); camera.lookAt(0,0,0);
                      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                      const dl = new THREE.DirectionalLight(0xffffff, 1);
                      dl.position.set(3,4,3); scene.add(dl);
                      scene.add(new THREE.GridHelper(10, 10, 0x333344, 0x222233));
                      const binColor = {json.dumps(bc)};
                      const mat = new THREE.MeshPhongMaterial({{color: new THREE.Color(binColor), shininess:80}});
                      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.55,0.65,1.4,32), mat);
                      body.position.set(0,0.7,0); scene.add(body);
                      const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.58,0.58,0.12,32), new THREE.MeshPhongMaterial({{color: new THREE.Color(binColor), shininess:100}}));
                      lid.position.set(0,1.46,0); scene.add(lid);
                      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.2,0.04,8,24), new THREE.MeshPhongMaterial({{color:0xffffff,opacity:0.6,transparent:true}}));
                      ring.position.set(0,0.7,0.56); ring.rotation.x=Math.PI/2; scene.add(ring);
                      const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.15,0.4,8), new THREE.MeshPhongMaterial({{color:0xffffff,opacity:0.85,transparent:true}}));
                      arrow.position.set(0,2.6,0); scene.add(arrow);
                      const pulse = new THREE.Mesh(new THREE.TorusGeometry(0.85,0.03,8,40), new THREE.MeshPhongMaterial({{color:new THREE.Color(binColor),opacity:0.5,transparent:true}}));
                      pulse.position.set(0,0.05,0); pulse.rotation.x=Math.PI/2; scene.add(pulse);
                      let t=0;
                      function animate(){{
                        requestAnimationFrame(animate); t+=0.02;
                        body.rotation.y=Math.sin(t*0.4)*0.2; lid.rotation.y=body.rotation.y; ring.rotation.y=body.rotation.y;
                        arrow.position.y=2.6+Math.sin(t*2)*0.15; pulse.scale.x=pulse.scale.y=1+Math.sin(t)*0.12;
                        renderer.render(scene,camera);
                      }}
                      animate();
                    }})();
                  </script>
                </div>"""
                st.components.v1.html(ar_html, height=230)
                if st.button("🔄  Scan Another", use_container_width=True):
                    st.session_state.pop("result", None); st.rerun()

        else:
            if "result" not in st.session_state:
                st.markdown("""
                <div style="background:#E8F0FE;border-radius:12px;padding:12px 16px;margin-bottom:12px;border-left:4px solid #1A73E8">
                  <div style="font-size:13px;color:#1A237E!important;font-weight:500">📸 Take a photo of your waste item, then click <b>Analyse</b> below.</div>
                </div>""", unsafe_allow_html=True)
                cam_img = st.camera_input("", label_visibility="collapsed", key="camera_input")
                if cam_img is not None:
                    img = Image.open(cam_img).convert("RGB")
                    st.image(img, caption="📸 Captured — analysing...", use_column_width=True)
                    with st.spinner("🔍 Running AI analysis ..."):
                        waste, conf, scores = predict_image(img)
                    st.session_state.result = {"waste":waste,"conf":conf,"scores":scores,"img":img}
                    st.session_state.history.append({"waste":waste,"conf":conf,"time":datetime.datetime.now().strftime("%H:%M:%S")})
                    st.rerun()
            else:
                st.image(st.session_state.result["img"], caption="📸 Captured Image", use_column_width=True)
                res  = st.session_state.result
                info = WASTE_INFO[res["waste"]]
                st.markdown(f"""
                <div class="cam-result-box">
                  <div style="display:flex;align-items:center;gap:14px">
                    <div style="font-size:40px">{info['icon']}</div>
                    <div>
                      <div style="font-size:20px;font-weight:800;color:#202124!important">{info['label']}</div>
                      <div style="font-size:13px;color:#5F6368!important">{info['bin_icon']} {info['bin']} &nbsp;·&nbsp;
                        <span style="color:#1A73E8!important;font-weight:700">{res['conf']}% confidence</span></div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄  Take Another Photo", use_container_width=True):
                    st.session_state.pop("result", None); st.rerun()

    with right:
        if "result" in st.session_state:
            res  = st.session_state.result
            info = WASTE_INFO[res["waste"]]
            st.markdown(f"""
            <div class="result-card">
              <div class="r-icon" style="background:{info['card_bg']}">{info['icon']}</div>
              <div class="r-name">{info['label']}</div>
              <div class="r-desc">{info['description'][:160]}...</div>
              <div class="r-conf-lbl">CONFIDENCE</div>
              <div class="r-conf-val">{res['conf']}%</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**📊 All Class Scores**")
            for cls, score in sorted(res["scores"].items(), key=lambda x: -x[1]):
                wi = WASTE_INFO.get(cls, WASTE_INFO["general"])
                is_top  = cls == res["waste"]
                bar_col = wi["bin_color"] if is_top else "#E8EAED"
                txt_col = wi["bin_color"] if is_top else "#5F6368"
                st.markdown(f"""
                <div style="margin-bottom:6px">
                  <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
                    <span style="color:#202124!important;font-weight:{'700' if is_top else '400'}">{wi['icon']} {wi['label']}</span>
                    <span style="color:{txt_col}!important;font-weight:700">{score}%</span>
                  </div>
                  <div style="background:#F1F3F4;border-radius:4px;height:6px">
                    <div style="background:{bar_col};border-radius:4px;height:6px;width:{int(score)}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        detected = st.session_state.get("result", {}).get("waste", None)
        seg_bins = [
            ("🟢","#1E8E3E","Organic / Biodegradable","Food waste, plants, paper towels",["organic"]),
            ("🔵","#1A73E8","Plastic / Paper / Metal","Bottles, boxes, clean paper, cans",["plastic","paper","metal"]),
            ("⚫","#5F6368","Glass","Jars, bottles, containers",["glass"]),
            ("🔴","#D93025","General / Residual","Mixed waste, hygiene, thermocol",["general"]),
        ]
        st.markdown('<div class="seg-panel"><div class="seg-title">🗂️ Smart Segregation</div>', unsafe_allow_html=True)
        for dot, color, title, sub, classes in seg_bins:
            ac = "active" if detected in classes else ""
            st.markdown(f"""
            <div class="seg-row {ac}">
              <div class="seg-dot" style="background:{color}"></div>
              <div class="seg-ml"><div class="seg-t">{title}</div><div class="seg-s">{sub}</div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tip = WASTE_INFO[st.session_state.result["waste"]]["eco_tip"] if "result" in st.session_state else "Proper waste segregation reduces landfill waste by up to 60% and helps create a cleaner, greener environment."
        st.markdown(f"""
        <div class="eco-card">
          <div class="eco-t">🌿 Eco Tip</div>
          <div class="eco-p">{tip}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — GUIDELINES
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Guidelines":
    st.markdown('<div class="status-bar">System Status: Active</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Waste Segregation Guidelines</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-bottom:20px"><div class="panel-title">📋 Quick Bin Reference</div>', unsafe_allow_html=True)
    for bin_name, color, bg, category, items in [
        ("🟢 Green Bin","#1E8E3E","#E6F4EA","Organic / Biodegradable","Food scraps, vegetable peels, fruit, garden waste, tea bags, coffee grounds"),
        ("🔵 Blue Bin","#1A73E8","#E8F0FE","Dry Recyclables — Plastic, Paper, Metal","Bottles, cans, newspapers, cardboard, magazines, office paper, aluminium foil"),
        ("⚫ Grey Bin","#5F6368","#F1F3F4","Glass","Glass bottles, jars, beverage containers — sorted by colour"),
        ("🔴 Red Bin","#D93025","#FCE8E6","General / Hazardous / Residual","Dirty wrappers, hygiene products, thermocol, batteries, broken items"),
    ]:
        st.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:16px 20px;
color:#202124 !important;">
          <div style="font-size:15px;font-weight:700;color:{color}!important">{bin_name}</div>
          <div style="font-size:12px;color:#5F6368!important;margin:2px 0 6px">{category}</div>
          <div style="font-size:13px;color:#3C4043!important">{items}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
<h3 style="color:#4ADE80;">
🗂️ Detailed Guidelines by Waste Type
</h3>
""", unsafe_allow_html=True)
    for wk, info in WASTE_INFO.items():
        with st.expander(f"{info['icon']}  {info['label']}   →   {info['bin_icon']} {info['bin']}"):
            c1,c2,c3 = st.columns([1,1,1], gap="medium")
            with c1:
                st.markdown(f"""
                <div style="background:{info['card_bg']};border-radius:14px;padding:18px;border-left:5px solid {info['bin_color']}">
                  <div style="font-size:16px;font-weight:700;color:{info['bin_color']}!important;margin-bottom:10px">{info['bin_icon']} {info['bin']}</div>
                  <div style="font-size:12px;font-weight:600;color:#5F6368!important;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Category</div>
                  <div style="font-size:13px;color:#3C4043!important;margin-bottom:10px">{info['category']}</div>
                  <div style="font-size:12px;font-weight:600;color:#5F6368!important;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Examples</div>
                  <div style="font-size:13px;color:#3C4043!important;margin-bottom:10px">{info['examples']}</div>
                  <div style="font-size:12px;font-weight:600;color:#5F6368!important;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">How Segregated</div>
                  <div style="font-size:13px;color:#3C4043!important">{info['segregation']}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("**✅ What TO DO**")
                for a in info["actions"]:
                    st.markdown(f'<div class="do-item"><span>✅ &nbsp;{a}</span></div>', unsafe_allow_html=True)
            with c3:
                st.markdown("**❌ What NOT to Do**")
                for d in info["do_not"]:
                    st.markdown(f'<div class="dont-item"><span>❌ &nbsp;{d}</span></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:#E6F4EA;border-radius:10px;padding:12px 14px;margin-top:10px;border-left:4px solid #1E8E3E">
                  <div style="font-size:11px;font-weight:700;color:#1E8E3E!important;margin-bottom:4px">💡 FACT</div>
                  <div style="font-size:12px;color:#1B3A1E!important">{info['fun_fact']}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ECO STORIES
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Eco Stories":
    st.markdown('<div class="status-bar">System Status: Active</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">🌍 Eco Stories & Upcycling Ideas</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#E6F4EA;color:#1B5E20 !important;border-radius:14px;padding:16px 22px;margin-bottom:24px;border-left:5px solid #1E8E3E">
      <div style="font-size:14px;color:#1B5E20!important;line-height:1.6">
        🌱 <b>Scan an item</b> in Waste Detection to get personalised eco-stories and upcycling ideas, or browse all categories below.
      </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        last_waste = st.session_state.history[-1]["waste"]
        info = WASTE_INFO.get(last_waste, WASTE_INFO["general"])
        st.markdown(f"### 📖 Last Scanned: {info['icon']} {info['label']}")
        st.markdown("""
        <div style="background:#FFF;border-radius:16px;color:#202124 !important;padding:22px 24px;border:1px solid #E8EAED;margin-bottom:20px">
          <div style="font-size:16px;font-weight:700;color:#202124!important;margin-bottom:16px">⏳ Life Cycle Journey</div>""",
                    unsafe_allow_html=True)
        lc_colors = ["#1A73E8","#34A853","#FBBC04","#EA4335","#9C27B0"]
        for i,(ico,title,desc) in enumerate(info["lifecycle"]):
            color = lc_colors[i % len(lc_colors)]
            st.markdown(f"""
            <div class="lc-step" style="border-color:{color}">
              <div class="lc-icon">{ico}</div>
              <div><div class="lc-title">{title}</div><div class="lc-desc">{desc}</div></div>
            </div>""", unsafe_allow_html=True)
        dec = info["decompose_years"]
        dec_str = (f"{int(dec*365)} days" if dec < 1 else f"{int(dec)} year{'s' if dec != 1 else ''}")
        st.markdown(f"""
        <div style="background:#FCE8E6;color:#B31412 !important;border-radius:12px;padding:14px 18px;margin-top:10px;border-left:5px solid #D93025">
          <div style="font-size:13px;color:#B31412!important">⚠️ <b>If sent to landfill</b> — takes <b>{dec_str}</b> to decompose.</div>
        </div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ### 💡 AI Upcycling Ideas for {info['label']}")
        up_cols = st.columns(min(len(info["upcycle"]), 3))
        for i, up in enumerate(info["upcycle"]):
            with up_cols[i % len(up_cols)]:
                diff_color = {"Easy":"#1E8E3E","Medium":"#F57C00","Mindset shift":"#1A73E8","Varies":"#9C27B0"}.get(up["difficulty"],"#5F6368")
                st.markdown(f"""
                <div class="up-card">
                  <div class="up-header">
                    <div class="up-emoji">{up['emoji']}</div>
                    <div>
                      <div class="up-title">{up['title']}</div>
                      <div class="up-meta">
                        <span style="background:{diff_color}22;color:{diff_color}!important;font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;margin-right:6px">{up['difficulty']}</span>
                        ⏱ {up['time']}
                      </div>
                    </div>
                  </div>
                  <div class="up-steps">{up['steps']}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("""
<h3 style="color:#FFFFFF; margin-top:10px;">
🗃️ Browse All Eco Stories
</h3>
""", unsafe_allow_html=True)
    tabs = st.tabs([f"{v['icon']} {v['label']}" for v in WASTE_INFO.values()])
    for tab, (wk, info) in zip(tabs, WASTE_INFO.items()):
        with tab:
            tc1, tc2 = st.columns([1,1], gap="large")
            with tc1:
                st.markdown(f"""
<div style="font-size:15px;font-weight:700;color:#FFFFFF;margin-bottom:14px">
⏳ Life Cycle of {info['label']}
</div>
""", unsafe_allow_html=True)
                
                lc_colors = ["#1A73E8","#34A853","#FBBC04","#EA4335","#9C27B0"]
                for i,(ico,title,desc) in enumerate(info["lifecycle"]):
                    color = lc_colors[i % len(lc_colors)]
                    st.markdown(f"""
                    <div class="lc-step" style="border-color:{color}">
                      <div class="lc-icon">{ico}</div>
                      <div><div class="lc-title">{title}</div><div class="lc-desc">{desc}</div></div>
                    </div>""", unsafe_allow_html=True)
                dec = info["decompose_years"]
                dec_str = (f"{int(dec*365)} days" if dec < 1 else f"{int(dec):,} year{'s' if dec != 1 else ''}")
                st.markdown(f"""
                <div style="background:#FCE8E6;color:#B31412 !important;border-radius:12px;padding:12px 16px;margin-top:10px;border-left:5px solid #D93025">
                  <div style="font-size:13px;color:#B31412!important">⚠️ <b>Landfill decomposition:</b> {dec_str}</div>
                </div>
                <div style="background:#E6F4EA;color:#1B5E20 !important;border-radius:12px;padding:12px 16px;margin-top:8px;border-left:5px solid #1E8E3E">
                  <div style="font-size:13px;color:#1B5E20!important">💡 {info['fun_fact']}</div>
                </div>""", unsafe_allow_html=True)
            with tc2:
                st.markdown("""<div style="font-size:15px;font-weight:700;color:#202124!important;margin-bottom:14px">💡 Upcycling Ideas</div>""", unsafe_allow_html=True)
                for up in info["upcycle"]:
                    diff_color = {"Easy":"#1E8E3E","Medium":"#F57C00","Mindset shift":"#1A73E8","Varies":"#9C27B0"}.get(up["difficulty"],"#5F6368")
                    st.markdown(f"""
                    <div class="up-card">
                      <div class="up-header">
                        <div class="up-emoji">{up['emoji']}</div>
                        <div>
                          <div class="up-title">{up['title']}</div>
                          <div class="up-meta">
                            <span style="background:{diff_color}22;color:{diff_color}!important;font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;margin-right:6px">{up['difficulty']}</span>
                            ⏱ {up['time']}
                          </div>
                        </div>
                      </div>
                      <div class="up-steps">{up['steps']}</div>
                    </div>""", unsafe_allow_html=True)