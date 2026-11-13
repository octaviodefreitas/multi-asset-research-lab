"""Asset lists for the app's pickers.

Optional energy futures: selectable in the research, portfolio and execution
tabs but NOT in the default universe or the frozen live config. Yahoo
continuous front-month series (contracts spliced at each roll, unadjusted).
Kept inside app/ so Streamlit Cloud hot-reloads it on every push.
"""
from data.loader import UNIVERSE

ENERGY = {
    "BZ=F": "Brent Crude futures (BZ=F)",
    "NG=F": "US Natural Gas futures (NG=F)",
}

ALL_ASSETS = {**UNIVERSE, **ENERGY}
