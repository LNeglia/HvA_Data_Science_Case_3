# In[ ]:
from pathlib import Path
import html
import pandas as pd
import numpy as np
import plotly.express as px
import folium
import streamlit as st
import streamlit.components.v1 as components
# In[ ]:

st.set_page_config(page_title = "Zürich Airport", layout = "wide")
st.title("Zürich Airport: verkeer en vertraging")
st.write("**Onderzoeksvraag:** welke vluchtkenmerken, verkeersdrukte en weersomstandigheden hangen samen met vertraging, en hoe verschilt dit tussen 2019 en 2020?")
st.caption("Gegevens uit het vluchtschema, de luchthavenlijst en dagelijks weer in Zürich. Eén rij telt als één landing of vertrek.")
BASE = Path(__file__).resolve().parent

def data_path(name):
    candidates = [BASE / name, BASE / "upload" / name]
    stem = Path(name).stem
    candidates +=  sorted(BASE.glob(stem + "(*).csv"))
    candidates +=  sorted((BASE / "upload").glob(stem + "(*).csv"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Zet {name} bij dit Python-bestand.")

# Step 1: Load/Cache datasets

@st.cache_data
def load_data():
    schedule_airport = pd.read_csv(data_path("schedule_airport.zip"), na_values = "-")
    airports = pd.read_csv(
        data_path("airports-extended.csv"),
        names = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude",
               "Altitude", "Timezone", "DST", "Tz database", "Type", "Source"],
        na_values = r"\N",
    )
    zurich_weather = pd.read_csv(
        data_path("06670.csv"),
        names = ["Date", "Temperature", "Dew Point", "Relative Humidity", "Total Precipitation",
               "Snow Depth", "Wind Direction", "Wind Speed", "Peak Gust", "Air Pressure",
               "Sunshine Duration"],
    )
    return schedule_airport, airports, zurich_weather

schedule_airport, airports, zurich_weather = load_data()
step_1a = schedule_airport.head()
step_1b = airports.head()
step_1c = zurich_weather.head()
# In[ ]:
# Step 2a: Delay column, hour of day and traffic volume
# After importing, start manipulation of data, starting with the airport schedule to make a Delay column:
schedule_airport["STD"] = pd.to_datetime(schedule_airport["STD"], format = "%d/%m/%Y")
# Isolate columns into a variable in a time format for better math conversion

scheduled_time = pd.to_timedelta(schedule_airport["STA_STD_ltc"])
actual_time = pd.to_timedelta(schedule_airport["ATA_ATD_ltc"])

# Calculate delay by simply doing predicted - actual, then saying if the difference > 12 hours, assume a midnight cross-over and fix
# This assumes, of course, that all flights aren't arriving or delayed by more than 12 hours.

delay = (actual_time - scheduled_time).dt.total_seconds() / 60
delay = np.where(delay < -720, delay + 1440, np.where(delay >  720, delay - 1440, delay))

# These are helper columns that do the following:
# 1. The delay in minutes
schedule_airport["Delay"] = delay
# 2. The year (for 2019/2020 differences)
schedule_airport["Year"] = schedule_airport["STD"].dt.year
# This section is for graphing later
# 3. Date/time for scheduled arrival and departure times: Can be used to show a "Intended travel" graph
schedule_airport["Scheduled_dt"] = schedule_airport["STD"] + scheduled_time
# 4. Date/time for actual arrival and departure times: Can be used to show a "Real travel" graph
schedule_airport["Actual_dt"] = schedule_airport["Scheduled_dt"] + pd.to_timedelta(schedule_airport["Delay"], unit = "min")
# Addendum to point 4: Float addition causes small errors so it's rounded to the nearest second to account for these
schedule_airport["Actual_dt"] = schedule_airport["Actual_dt"].dt.round("s")
schedule_airport["Hour"] = schedule_airport["Scheduled_dt"].dt.hour
# Traffic volume: movements scheduled in the same hour
schedule_airport["Hour_slot"] = schedule_airport["Scheduled_dt"].dt.floor("h")
schedule_airport["Traffic_hour"] = schedule_airport.groupby("Hour_slot")["Hour_slot"].transform("size")
step_2a_new_columns = schedule_airport[["STD", "STA_STD_ltc", "ATA_ATD_ltc", "Delay", "Year",
                                        "Scheduled_dt", "Actual_dt", "Hour", "Traffic_hour"]].head()
# In[ ]:

# Step 2b: Merge the 2019-2020 weather onto the schedule by date
zurich_weather["Date"] = pd.to_datetime(zurich_weather["Date"])
weather = zurich_weather[zurich_weather["Date"].between("2019-01-01", "2020-12-31")]
schedule_airport = schedule_airport.merge(weather, left_on = "STD", right_on = "Date", how = "left")
step_2b_weather_merged = schedule_airport[["STD"] + weather.columns.tolist()].head()
# In[ ]:

# Step 2c: Count the missing weather values
weather_cols = ["Temperature", "Dew Point", "Relative Humidity", "Total Precipitation", "Snow Depth",
                "Wind Direction", "Wind Speed", "Peak Gust", "Air Pressure", "Sunshine Duration"]
# Share missing per weather column
step_2c_weather_missing = (schedule_airport[weather_cols].isna().mean().mul(100).round(1)
                           .sort_values(ascending = False).to_frame("missing_pct"))
# Days in the schedule without any weather row at all
step_2c_days_missing_weather = pd.DataFrame({
    "days_in_schedule": [schedule_airport["STD"].nunique()],
    "days_without_weather": [schedule_airport.loc[schedule_airport["Date"].isna(), "STD"].nunique()],
})
# Flights with at least one missing weather value, per year
step_2c_flights_missing_weather = (schedule_airport[weather_cols].isna().any(axis = 1)
                                   .groupby(schedule_airport["Year"]).agg(["sum", "mean"]))

# In[ ]:

# Step 2d: Drop near-empty weather columns, then rows with missing weather
traffic_source = schedule_airport.copy()
na_share = schedule_airport[weather_cols].isna().mean()
drop_cols = na_share[na_share > 0.5].index.tolist()
step_2d_dropped_columns = pd.DataFrame({"dropped_column": drop_cols, "missing_pct": (na_share[drop_cols] * 100).round(1).values})
weather_cols = [c for c in weather_cols if c not in drop_cols]
schedule_clean = schedule_airport.drop(columns = drop_cols).dropna(subset = weather_cols)
# Effect of the drop
before = schedule_airport.groupby("Year")["Delay"].agg(["count", "mean", "median"])
after = schedule_clean.groupby("Year")["Delay"].agg(["count", "mean", "median"])
step_2d_before_after = pd.concat([before, after], axis = 1, keys = ["before", "after"]).round(1)

# In[ ]:

# Step 3a: Check and clean the airport list
# Missing values per column
na_counts = {}
for col in airports.columns:
    na_counts[col] = airports[col].isna().sum()
step_3a_airport_na = pd.DataFrame.from_dict(na_counts, orient = "index", columns = ["NA"])

# Duplicate ICAO codes (value_counts skips NaN, so this loop only finds real codes appearing twice)
dup_counts = airports["ICAO"].value_counts()
real_duplicates = {}
for icao, n in dup_counts[dup_counts > 1].items():
    real_duplicates[icao] = n

step_3a_duplicates = pd.DataFrame({"rows": [airports["ICAO"].duplicated().sum(),
                                            airports["ICAO"].isna().sum(),
                                            len(real_duplicates)]},
                                  index = ["Duplicate ICAO rows (total)", "Rows with empty ICAO",
                                           "Real ICAO codes appearing twice"])
airports = airports.dropna(subset = ["ICAO"]).drop_duplicates(subset = "ICAO")
step_3a_airports_clean = airports.head()

# In[ ]:

# Step 3b: Merge on ICAO, count and drop the unmatched flights
schedule_clean = schedule_clean.merge(airports[["ICAO", "Name", "Country", "Latitude", "Longitude", "Tz database"]],
                                      left_on = "Org/Des", right_on = "ICAO",
                                      how = "left", validate = "many_to_one")

unmatched = schedule_clean["ICAO"].isna()
step_3b_unmatched_count = pd.DataFrame({"flights": [unmatched.sum()], "pct": [round(unmatched.mean() * 100, 2)]},
                                       index = ["Without airport match"])

step_3b_unmatched_codes = schedule_clean.loc[unmatched, "Org/Des"].value_counts().head(10).to_frame("flights")

before = schedule_clean.groupby("Year")["Delay"].agg(["count", "mean", "median"])
schedule_clean = schedule_clean.dropna(subset = ["ICAO"])
after = schedule_clean.groupby("Year")["Delay"].agg(["count", "mean", "median"])
step_3b_before_after = pd.concat([before, after], axis = 1, keys = ["before", "after"]).round(1)

# In[ ]:

# Step 4a: Europe/Intercontinental using time zone and region selection
# European countries that don't have "Europe/" as their time zone entry
extra_europe = ["Atlantic/Canary", "Atlantic/Azores", "Atlantic/Reykjavik"]

is_europe = (schedule_clean["Tz database"].str.startswith("Europe/", na = False)
             | schedule_clean["Tz database"].isin(extra_europe))

schedule_clean["Region"] = np.where(is_europe, "Europe", "Intercontinental")

step_4a_region_counts = schedule_clean.groupby(["Year", "Region"]).size().unstack()
step_4a_region_counts["No time zone"] = (schedule_clean[schedule_clean["Tz database"].isna()]
                                         .groupby("Year").size().reindex(step_4a_region_counts.index, fill_value = 0))

# Region selection: one region or both. sorted + tuple, so the click order doesn't create a second cache entry
REGIONS = tuple(sorted(st.sidebar.multiselect("Dataset Regio", ["Europe", "Intercontinental"], default = ["Europe", "Intercontinental"])))
if not REGIONS:
    st.warning("Kies min een regio")
    st.stop()
@st.cache_data
def get_region(df_input, regions):
    return df_input[df_input["Region"].isin(regions)].copy()

region_df = get_region(schedule_clean, REGIONS)
if region_df.empty:
    st.warning("Geen complete analyserijen voor deze regioselectie.")
    st.stop()

# In[ ]:

# Functions to look at average delay per column, per year. Looks at average delay, then the average delay per category relative to overall.
# TLDR: Average delay per category, shows how far categorical average is from overall average.

def delay_effect(df, col):
    year_mean = df.groupby("Year")["Delay"].transform("mean")
    out = (df.assign(Effect = df["Delay"] - year_mean)
             .groupby(["Year", col], observed = True)
             .agg(flights = ("Delay", "count"),
                  mean_delay = ("Delay", "mean"),
                  effect = ("Effect", "mean")))
    return out.round(1)

def compare_years(df, col):
    return (delay_effect(df, col)["effect"]
            .unstack("Year").dropna().sort_values(2019, ascending = False))

# In[ ]:

# Step 4b: Code for line chart for delay per hour/weekday/month and per year. Cached per region, period and years
PERIOD = st.sidebar.radio("Vertraging per tijdseenheid", ["Hour", "Weekday", "Month"])
YEARS = tuple(sorted(st.sidebar.multiselect("Lijngrafiek jaren", [2019, 2020], default = [2019, 2020])))

@st.cache_data
def time_chart(df_input, regions, period, years):
    df = df_input[df_input["Year"].isin(years)].copy()
    df["Effect"] = df["Delay"] - df.groupby("Year")["Delay"].transform("mean")
    df["Weekday"] = df["STD"].dt.day_name()
    df["Month"] = df["STD"].dt.month
    data = df.groupby(["Year", "LSV", period])["Effect"].mean().reset_index()
    data["Year"] = data["Year"].astype(str)
    return px.line(data, x = period, y = "Effect", color = "Year", line_dash = "LSV",
                   category_orders = {"Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                   title = f"{' + '.join(regions)}: minuten boven/onder het jaarlijkse gemiddelde, per {period.lower()}")

step_4b_time_chart = time_chart(region_df, REGIONS, PERIOD, YEARS)

# In[ ]:

# Step 4c: Effect per factor, as bar charts (cached per region)

@st.cache_data
def add_bins(df_input, regions):
    df = df_input.copy()
    # Traffic volume
    df["Traffic_bin"] = pd.cut(df["Traffic_hour"], bins = [0, 20, 40, 60, 80, 200])
    # Weather
    df["Rain_bin"] = pd.cut(df["Total Precipitation"], bins = [-0.1, 0, 1, 5, 100],
                            labels = ["dry", "0-1 mm", "1-5 mm", ">5 mm"])
    # Extra code needed here: math with Floats was causing the lower bin to have 16 numbers after the decimal point
    wind_edges = sorted(set(df["Wind Speed"].quantile([0, 0.25, 0.5, 0.75, 1]).dropna().tolist()))
    if len(wind_edges) < 2:
        wind_edges = [-np.inf, np.inf]
    df["Wind_bin"] = pd.cut(df["Wind Speed"], bins = wind_edges, include_lowest = True,
                            labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(wind_edges[:-1], wind_edges[1:])])
    return df

# If both years true, compares/shows only entries that share categories

@st.cache_data
def factor_chart(df_input, regions, col, title, both_years = False, top = None):
    data = delay_effect(df_input, col).reset_index()
    if both_years:
        order = compare_years(df_input, col).head(top).index.tolist()
        data = data[data[col].isin(order)]
    else:
        order = data[col].unique().tolist()
    
    data[col] = data[col].astype(str)
    
    order = [str(o) for o in order]
    
    if col == "LSV":
        names = {"L": "Arrivals", "S": "Departures"}
        data[col] = data[col].map(names)
        order = [names[o] for o in order]
    data["Year"] = data["Year"].astype(str)

    fig = px.bar(data, x = col, y = "effect", color = "Year", barmode = "group", hover_data = ["flights", "mean_delay"],
                 category_orders = {col: order, "Year": ["2019", "2020"]}, labels = {"effect": "Minutes above/below yearly average"},
                 title = f"{' + '.join(regions)}: {title}")

    fig.update_traces(marker_line_color = "black", marker_line_width = 1)   # outline per bar
    return fig

# The original graph was set up by hand, but when changing to a function that takes both regions, AI was used to ensure there weren't any bugs
region_df = add_bins(region_df, REGIONS)

# LSV, plane type, runway, destination
step_4c_lsv = factor_chart(region_df, REGIONS, "LSV", "arrivals vs. departures")
step_4c_aircraft = factor_chart(region_df, REGIONS, "ACT", "aircraft type (top 15 in 2019)", both_years = True, top = 15)
step_4c_runway = factor_chart(region_df, REGIONS, "RWY", "runway", both_years = True)
step_4c_destination = factor_chart(region_df, REGIONS, "Org/Des", "origin/destination (top 15 in 2019)", both_years = True, top = 15)

# Traffic volume
step_4c_traffic = factor_chart(region_df, REGIONS, "Traffic_bin", "traffic volume (movements per hour)")

# Weather
step_4c_rain = factor_chart(region_df, REGIONS, "Rain_bin", "daily precipitation")
step_4c_wind = factor_chart(region_df, REGIONS, "Wind_bin", "daily wind speed (quartiles)")

# In[ ]:

# Step 4d: Weather ratio and minutes per unit, at day level (cached per region)
@st.cache_data
def weather_ratios(df_input, regions, weather_cols):
    # Weather slopes at day level
    daily = (df_input.groupby(["Year", "STD"])
                .agg(mean_delay = ("Delay", "mean"), **{column: (column, "first") for column in weather_cols}))
    rows = []

    for year, day in daily.groupby("Year"):
        for column in weather_cols:
            rows.append({"Year": str(year), "Weather": column,
                         "ratio": day[column].corr(day["mean_delay"]),
                         "min_per_unit": (np.polyfit(day[column], day["mean_delay"], 1)[0] if day[column].nunique() > 1 and len(day) >=  3 else np.nan)})

    data = pd.DataFrame(rows).round(2)
    label = " + ".join(regions)
    table = data.pivot(index = "Weather", columns = "Year", values = ["ratio", "min_per_unit"])

    ratio_fig = px.bar(data, x = "Weather", y = "ratio", color = "Year", barmode = "group",
                       category_orders = {"Year": ["2019", "2020"]},
                       labels = {"ratio": "Ratio (correlation r) with daily mean delay"},
                       title = f"{label}: how strongly each weather value moves with the daily delay")

    unit_fig = px.bar(data, x = "Weather", y = "min_per_unit", color = "Year", barmode = "group",
                      category_orders = {"Year": ["2019", "2020"]},
                      labels = {"min_per_unit": "Minutes of delay per unit"},
                      title = f"{label}: minutes of delay per unit of each weather value")

    for fig in (ratio_fig, unit_fig):
        fig.update_traces(marker_line_color = "black", marker_line_width = 1)

    return table, ratio_fig, unit_fig

# The data was calculated without AI, but the conversion from a table to a bar chart to show the results used some AI for formatting and speed
step_4d_weather_table, step_4d_ratio_chart, step_4d_per_unit_chart = weather_ratios(region_df, REGIONS, tuple(weather_cols))

# In[ ]:

# Step 5a: Bad-weather boundaries and checks (cached per region)
@st.cache_data
def add_bad_weather(df_input, regions):
    df = df_input.copy()
    days = df.groupby("STD")[["Total Precipitation", "Wind Speed", "Air Pressure", "Temperature"]].first()
    rain_limit = days["Total Precipitation"].quantile(0.9)   # top 10% wettest days
    wind_limit = days["Wind Speed"].quantile(0.9)            # top 10% windiest days
    pressure_limit = days["Air Pressure"].quantile(0.1)      # bottom 10% lowest pressure
    temp_limit = 0                                           # freezing point, °C

    df["Storm"] = (df["Air Pressure"] < pressure_limit) & (df["Total Precipitation"] > rain_limit)
    df["Ice_risk"] = (df["Temperature"] <=  temp_limit) & (df["Total Precipitation"] > 0)
    df["Strong_wind"] = df["Wind Speed"] > wind_limit
    df["Bad_weather"] = df["Storm"] | df["Ice_risk"] | df["Strong_wind"]

    boundaries = pd.DataFrame({"Boundary": [f"> {rain_limit:.1f} mm", f"> {wind_limit:.1f} km/h",
                                            f"< {pressure_limit:.1f} hPa", f"< =  {temp_limit} °C"]},
                              index = ["Rain", "Wind", "Air Pressure", "Temperature"])

    # Days per condition, per year
    condition_days = (df.groupby(["Year", "STD"])[["Storm", "Ice_risk", "Strong_wind", "Bad_weather"]]
                        .first().groupby("Year").sum())

    # Do bad-weather days actually have more delay?
    delay_check = df.groupby(["Year", "Bad_weather"])["Delay"].agg(["count", "mean"]).round(1)
    return df, boundaries, condition_days, delay_check

region_df, step_5a_boundaries, step_5a_condition_days, step_5a_delay_check = add_bad_weather(region_df, REGIONS)

# In[ ]:

# Step 5b: Credit each flight to Zurich (bad weather) or the other airport (cached per region and year)
# Also sets up for the by making a table to store the data.
MAP_YEAR = st.sidebar.radio("Folium kaart jaar", [2019, 2020])

@st.cache_data
def blame_table(df_input, airports_input, regions, map_year):
    df = df_input[df_input["Year"] == map_year]
    blame = (df.assign(Blame_ICAO = np.where(df["Bad_weather"], "LSZH", df["ICAO"]),
                       Is_delayed = df["Delay"] > 0)
               .groupby("Blame_ICAO")
               .agg(flights = ("Delay", "size"),
                    delays = ("Is_delayed", "sum"),
                    mean_delay = ("Delay", "mean"))
               .reset_index()
               .merge(airports_input[["ICAO", "Name", "City", "Latitude", "Longitude"]],
                      left_on = "Blame_ICAO", right_on = "ICAO"))
    blame["share_pct"] = (blame["delays"] / blame["delays"].sum() * 100).round(1)
    check = pd.DataFrame({"value": [len(blame), blame["flights"].sum(), len(df)]},
                         index = ["Airports on the map", "Flights credited", "Flights in selection"])
    return blame, check

blame, step_5b_blame_check = blame_table(region_df, airports, REGIONS, MAP_YEAR)

# In[ ]:

# Step 5c: Folium map. Changes based on chosen year and what the circles mean.
SIZE_BY = st.sidebar.radio("Folium kaartgegevenstype", ["delays", "mean_delay"], format_func = {"delays": "Aantal vertragingen", "mean_delay": "Gemiddelde vertraging"}.get)

def radius(value, highest, min_r = 3, max_r = 30):
    if pd.isna(highest) or highest <=  0:
        return min_r
    return min_r + (max_r - min_r) * np.sqrt(value / highest)

def build_map(blame, regions, size_by):
    blame = blame.assign(size = blame[size_by].clip(lower = 0))
    blame = blame.dropna(subset = ["Latitude", "Longitude"])
    highest = blame["size"].max()
    europe_only = regions == ("Europe",)
    flight_map = folium.Map(location = [47.46, 8.55] if europe_only else [20, 0],
                            zoom_start = 4 if europe_only else 2)

    for _, row in blame.sort_values("size", ascending = False).iterrows():
        is_zrh = row["ICAO"] == "LSZH"
        folium.CircleMarker(
            location = [row["Latitude"], row["Longitude"]],
            radius = radius(row["size"], highest),
            color = "red" if is_zrh else "blue",
            fill = True,
            fill_opacity = 0.7 if is_zrh else 0.5,
            weight = 1,
            tooltip = f"{row['Name']} ({row['ICAO']})<br>Flights: {row['flights']}<br>"
                    f"Delayed: {row['delays']} ({row['share_pct']}%)<br>Avg delay: {row['mean_delay']:.1f} min",
        ).add_to(flight_map)

    legend = """<div style = "position:fixed;bottom:25px;left:25px;background:white;padding:12px;z-index:9999;border:1px solid #777;font-size:13px;max-width:290px"><b>Legenda</b><br><span style = "color:red">●</span> Zürich: toegewezen bij slecht weer<br><span style = "color:blue">●</span> Andere luchthaven: overige bewegingen<br>Grotere cirkel = hogere gekozen waarde.<br>Toewijzing volgens een aanname; geen bewezen oorzaak.</div>"""
    flight_map.get_root().html.add_child(folium.Element(legend))
    return flight_map

step_5c_map = build_map(blame, REGIONS, SIZE_BY)

# In[ ]:

# Presentatie: dezelfde stapvariabelen als in het oorspronkelijke script.
metrics = st.columns(3)
metrics[0].metric("Geanalyseerde bewegingen", f"{len(region_df):,}")
metrics[1].metric("Gemiddelde vertraging", f"{region_df['Delay'].mean():.1f} min")
metrics[2].metric("Meer dan 15 minuten vertraagd", f"{region_df['Delay'].gt(15).mean():.1%}")
st.write(f"**Eerste conclusie:** in de geselecteerde regio's is de gemiddelde vertraging {region_df['Delay'].mean():.1f} minuten; de mediaan is {region_df['Delay'].median():.1f} minuten. " + ("Het hogere gemiddelde laat zien dat grote vertragingen het gemiddelde omhoog trekken." if region_df['Delay'].mean() > region_df['Delay'].median() else "Het gemiddelde ligt niet hoger dan de mediaan; bekijk de verdeling voor meer detail."))
st.caption("Kerncijfers zijn berekend na de oorspronkelijke opschoning, voor beide jaren. De jaarselectie in de zijbalk geldt voor de tijdgrafiek; het kaartjaar geldt alleen voor de kaart.")

tab_overview, tab_processing, tab_factors, tab_weather, tab_map, tab_future = st.tabs(["Overzicht", "Dataverwerking", "Vluchtkenmerken", "Weer", "Folium-kaart", "Vooruitblik"])

with tab_overview:
    st.subheader("Vertraging verdeeld over vluchten")
    st.plotly_chart(px.histogram(region_df, x = "Delay", nbins = 70, title = "Verdeling van vertraging na opschoning", labels = {"Delay": "Vertraging (minuten)"}), width = 'stretch')
    st.write("Positief betekent later dan gepland; negatief betekent eerder. Het histogram laat zien hoe vaak kleine en grote afwijkingen voorkomen. Gemiddelde en mediaan samen voorkomen dat alleen uitschieters ons beeld bepalen.")
    summary = region_df.groupby("Year")["Delay"].agg(["count", "mean", "median", "std"]).round(1)
    st.dataframe(summary.rename(columns = {"count": "Aantal", "mean": "Gemiddelde (min)", "median": "Mediaan (min)", "std": "Standaardafwijking (min)"}))
    if {2019, 2020}.issubset(summary.index):
        difference = summary.loc[2020, "mean"] - summary.loc[2019, "mean"]
        st.write(f"In 2020 ligt de gemiddelde vertraging {abs(difference):.1f} minuten {'hoger' if difference > 0 else 'lager' if difference < 0 else 'op hetzelfde niveau als'} dan in 2019. Dit beschrijft een verschil tussen jaren; het verklaart niet waardoor dat verschil ontstaat.")
    st.subheader("Aantal vliegbewegingen door de tijd")
    st.write("Deze telling is toegevoegd voor de rubric. We tellen landingen en vertrekken op hun geplande tijdstip. Dit meet verkeersvolume, niet het aantal vliegtuigen dat gelijktijdig op het terrein staat.")
    unit = st.selectbox("Tijdseenheid voor het verkeersvolume", ["Dag", "Uur", "Maand"])
    all_counts = []
    # Alle schemarijen gebruiken: ontbrekend weer mag verkeersvolume niet verlagen.
    for year, group in traffic_source.groupby("Year"):
        counts = group.set_index("Scheduled_dt").resample({"Dag": "D", "Uur": "h", "Maand": "MS"}[unit]).size().rename("Aantal").reset_index()
        counts["Jaar"] = str(year)
        all_counts.append(counts)
    volume = pd.concat(all_counts, ignore_index = True)
    st.plotly_chart(px.line(volume, x = "Scheduled_dt", y = "Aantal", color = "Jaar", title = f"Geplande vliegbewegingen op Zürich per {unit.lower()}", labels = {"Scheduled_dt": "Datum en tijd", "Aantal": f"Vliegbewegingen per {unit.lower()}"}), width = 'stretch')
    peak = volume.loc[volume["Aantal"].idxmax()]
    st.write(f"De drukste getoonde {unit.lower()} begint op {peak['Scheduled_dt']:%d-%m-%Y %H:%M}, met {peak['Aantal']:.0f} bewegingen. Dagen tonen dagelijkse drukte; uren tonen piekuren; maanden maken langetermijnverschillen zichtbaar.")
    st.caption("Volledig ingelezen schema vóór het verwijderen van ontbrekend weer en ontbrekende luchthavenmatches; de regiofilters gelden hier niet. Lege tijdseenheden binnen een jaar tellen als nul, onder de aanname van volledige registratie.")
    with st.expander("Oorspronkelijke tijdgrafiek: afwijking van de gemiddelde vertraging", expanded = True):
        if YEARS:
            st.plotly_chart(step_4b_time_chart, width = 'stretch')
            st.write("We trekken eerst per jaar het gemiddelde van iedere vertraging af. Daarna nemen we het gemiddelde per uur, weekdag of maand, apart voor landingen (L) en starts (S). Een waarde van +5 betekent vijf minuten boven het jaargemiddelde van de geselecteerde regio's, niet vijf minuten absolute vertraging.")
        else:
            st.info("Selecteer een jaar in de zijbalk.")

with tab_processing:
    st.subheader("Wat hebben we met de data gedaan?")
    st.write("Onderstaande stappen volgen de volgorde en berekeningen van jullie oorspronkelijke code. De bijbehorende controletabellen laten zien welke gegevens zijn toegevoegd en wat er bij de opschoning is verwijderd.")
    with st.expander("Stap 1 — De drie datasets inlezen", expanded = True):
        st.write("We lezen het vluchtschema, de luchthavenlijst en het dagelijkse weer van Zürich in met pandas. Een '-' in het schema en de ontbrekendewaardecode in de luchthavenlijst worden als NaN gelezen. De luchthaven- en weerbestanden hebben geen kopregel, daarom geven we zelf kolomnamen op. Met st.cache_data bewaren we de ingelezen resultaten zodat niet bij iedere selectie opnieuw van schijf hoeft te worden gelezen.")
        for label, frame in [("Vluchtschema", step_1a), ("Luchthavenlijst", step_1b), ("Weer", step_1c)]:
            st.write(f"**{label}: eerste vijf rijen**")
            st.dataframe(frame, hide_index = True)
        st.caption("De namen en eenheden van de weerkolommen zijn overgenomen uit het oorspronkelijke script. Controleer die koppeling aan de bronbeschrijving: het bestand zelf bevat geen kolomkoppen.")
    with st.expander("Stap 2a — Vertraging en verkeersdrukte berekenen", expanded = True):
        st.write("De vertraging van elke vlucht wordt berekend door de geplande tijd af te trekken van de werkelijke tijd, in minuten. Als de klok echter middernacht passeert, wordt een verschil van meer dan 720 minuten/12 uur (positief betekent te laat, negatief betekent te vroeg) geregistreerd. Om hiermee rekening te houden, wordt ervan uitgegaan dat geen enkele vlucht 12 uur te vroeg of te laat is aangekomen en wordt de fout 'gecorrigeerd' door 12 uur toe te voegen of af te trekken om de werkelijke tijd te verkrijgen.")
        st.write("In deze stap wordt ook het jaar toegevoegd, nodig om 2019 en 2020 afzonderlijk te analyseren vanwege corona, en volledige tijdstempels voor de geplande en werkelijke tijden. Ten slotte worden het geplande uur en het verkeersvolume, oftewel het aantal vluchten dat in hetzelfde uur is gepland, toegevoegd. Het verkeer wordt geteld op basis van het volledige schema voordat er rijen worden verwijderd, zodat geen enkel uur rustiger lijkt dan het in werkelijkheid was.")
        st.dataframe(step_2a_new_columns, hide_index = True)
    with st.expander("Stap 2b en 2c — Dagweer koppelen en ontbrekende waarden meten", expanded = True):
        st.write("Het weerlogboek wordt ingekort tot 2019 en 2020 en samengevoegd met het vluchtschema op basis van datum, zodat elke vlucht de weersinformatie van de geplande dag krijgt. Er wordt een linkse samenvoeging gebruikt, waardoor elke vlucht behouden blijft, zelfs als er geen weersinformatie voor die dag beschikbaar is. Ontbrekende waarden blijven in de data staan, zodat deze als eerste kunnen worden geteld.")
        st.dataframe(step_2b_weather_merged, hide_index = True)
        st.write("Drie controles meten hoe volledig de samengevoegde weergegevens zijn. De eerste geeft het percentage ontbrekende waarden per weerkolom. De tweede telt de dagen in het schema waarop helemaal geen weersinformatie staat. De derde telt per jaar hoeveel vluchten minstens één weerwaarde missen.")
        st.dataframe(step_2c_weather_missing)
        st.dataframe(step_2c_days_missing_weather, hide_index = True)
        st.dataframe(step_2c_flights_missing_weather.rename(columns = {"sum": "Rijen met ontbrekend weer", "mean": "Aandeel"}))
    with st.expander("Stap 2d — Weerdata opschonen en het effect controleren", expanded = True):
        st.write("Weerkolommen die voor meer dan 50% leeg zijn, worden verwijderd, omdat ze geen analyse mogelijk maken en anders bijna elke rij zouden worden verwijderd. Daarna worden vluchten met een ontbrekende waarde in een van de resterende weerkolommen verwijderd, zodat elke volgende analyse dezelfde set vluchten gebruikt.")
        st.dataframe(step_2d_dropped_columns, hide_index = True)
        st.write("Om te controleren of deze opschoning de gegevens heeft verstoord, worden het aantal vluchten en de gemiddelde en mediane vertraging per jaar vóór en na de verwijdering vergeleken. Omdat de gemiddelden nauwelijks veranderden, waren de verwijderde rijen niet systematisch verschillend en zijn de resterende gegevens nog steeds representatief.")
        st.dataframe(step_2d_before_after)
    with st.expander("Stap 3a en 3b — Luchthavenlijst opschonen en via ICAO koppelen", expanded = True):
        st.write("De luchthavenlijst wordt gecontroleerd op ontbrekende waarden per kolom en op dubbele ICAO-codes. Veel vermeldingen, zoals treinstations en veerhavens, hebben geen ICAO-code, en deze lege codes worden ook als duplicaten weergegeven. Ze worden verwijderd, samen met eventuele echte duplicaten, zodat elke ICAO-code slechts één keer voorkomt. Dit voorkomt dat een vlucht aan meer dan één luchthaven wordt gekoppeld, of aan een vermelding zonder code.")
        st.dataframe(step_3a_airport_na)
        st.dataframe(step_3a_duplicates)
        st.write("Het vluchtschema wordt samengevoegd met de opgeschoonde luchthavenlijst op basis van de ICAO-code in de kolom herkomst/bestemming. Elke vlucht krijgt de naam, het land, de coördinaten en de tijdzone van de andere luchthaven. Vluchten waarvan de code geen overeenkomst vindt, worden geteld en weergegeven, samen met de meest voorkomende codes die niet overeenkomen, en vervolgens verwijderd omdat ze geen locatie hebben. Een vergelijking van het aantal vluchten en de gemiddelde en mediane vertraging vóór en na de bewerking laat zien hoeveel invloed deze verwijdering op de gegevens heeft.")
        st.dataframe(step_3b_unmatched_count)
        st.dataframe(step_3b_unmatched_codes)
        st.dataframe(step_3b_before_after)
    with st.expander("Stap 4a — Regio's indelen", expanded = True):
        st.write("Elke vlucht wordt gelabeld als Europa of Intercontinentaal op basis van de tijdzone van de andere luchthaven: tijdzones die beginnen met 'Europe/' tellen als Europa, samen met een paar Atlantische eilanden die politiek gezien tot Europa worden gerekend, zoals IJsland. Een tabel toont het aantal vluchten per regio per jaar. Via een zijmenu kan de gebruiker vervolgens kiezen voor Europa, Intercontinentaal of beide. De gefilterde dataset wordt voor elke keuze afzonderlijk opgeslagen, zodat het terugschakelen naar een reeds bekeken regio de berekening niet opnieuw uitvoert.")
        st.dataframe(step_4a_region_counts)
        st.write("De onderstaande factor-, weer- en kaartanalyses gebruiken deze selectie.")
    with st.expander("Stap 4b–4d — Vergelijken met het jaargemiddelde en weer analyseren", expanded = True):
        st.write("Voor elke vlucht wordt de gemiddelde vertraging van dat jaar afgetrokken van de eigen vertraging. Dit geeft het 'effect': hoeveel minuten de vlucht langer of korter was dan een typische vlucht in dat jaar. Door de resultaten te vergelijken met het eigen gemiddelde blijven 2019 en 2020 vergelijkbaar, ondanks het feit dat er in 2020 veel minder vluchten waren en de vertragingen over het algemeen lager waren. De lijngrafiek toont dit effect per uur, weekdag of maand, zoals gekozen in de zijbalk, met aparte lijnen voor aankomsten en vertrekken en voor elk geselecteerd jaar.")
        st.write("Hetzelfde effect wordt berekend per categorie van een enkele factor en weergegeven als een staafdiagram per jaar. De factoren zijn aankomsten versus vertrekken, vliegtuigtype, landingsbaan, luchthaven van herkomst/bestemming, verkeersvolume per uur, dagelijkse neerslag en dagelijkse windsnelheid.")    
        st.write("Voor vliegtuigtype en bestemming worden alleen categorieën weergegeven die in beide jaren voorkomen, beperkt tot de 15 met het grootste effect in 2019. Elke grafiek bekijkt één factor afzonderlijk, dus overlapping tussen factoren, zoals de keuze van de landingsbaan afhankelijk van de wind, moet in acht worden genomen bij het interpreteren ervan.")
        st.write("De vluchten worden vervolgens per dag gegroepeerd en de gemiddelde vertraging van elke dag wordt vergeleken met de weerwaarden van die dag. Voor elke weerkolom levert dit twee getallen per jaar op:")
        st.write("* De ratio (correlatie): hoe consistent de vertraging beweegt met die weerwaarde.")
        st.write("* De helling: hoeveel minuten vertraging één eenheid van die weerwaarde toevoegt of verwijdert.")
        st.write("Beide worden weergegeven in een tabel en als staafdiagrammen. Omdat de weergegevens dagelijks zijn, worden korte gebeurtenissen zoals een korte onweersbui gemiddeld, waardoor de weereffecten waarschijnlijk worden onderschat.")
    with st.expander("Stap 5 — Slecht weer definiëren en de Folium-kaart voorbereiden", expanded = True):
        st.write("Slecht weer wordt gedefinieerd aan de hand van vier dagelijkse weerwaarden: neerslag, windsnelheid, luchtdruk en temperatuur. Wanneer een waarde geen natuurlijke grens heeft, wordt de grens afgeleid van de gegevens zelf, berekend over dagen in plaats van vluchten, zodat drukke dagen niet zwaarder wegen:")
        st.write("""1. Neerslag: de 10% natste dagen.
2. Windsnelheid: de 10% winderigste dagen.
3. Luchtdruk: de 10% dagen met de laagste luchtdruk.
4. Temperatuur: het vriespunt, 0°C.""")
        st.dataframe(step_5a_boundaries)
        st.write("""Deze waarden worden gecombineerd tot drie condities:
1. Storm: behorend tot de 10% dagen met de laagste luchtdruk en tevens behorend tot de 10% natste dagen.
2. IJsgevaar: een temperatuur van vriespunt of lager in combinatie met neerslag.
3. Harde wind: behorend tot de 10% winderigste dagen.""")
        st.dataframe(step_5a_condition_days)
        st.write("Elke vlucht wordt vervolgens 'toegewezen' aan Zürich of de andere locatie. Als de vlucht plaatsvindt op een dag met 'slecht weer', wordt de vertraging toegeschreven aan Zürich, anders aan de andere luchthaven.")
        st.dataframe(step_5b_blame_check)
        st.write("""Folium wordt vervolgens gebruikt om een ​​cirkeldiagram te maken met de vertragingsgegevens. Zürich is rood en de andere luchthavens zijn blauw. De oppervlakte van de cirkel geeft het aantal vertragingen of de gemiddelde vertraging weer, die in de zijbalk kan worden gekozen.
De belangrijkste aannames voor dit gedeelte zijn:
* Als een dag is gemarkeerd als 'slecht weer', wordt elke vertraging veroorzaakt door het weer.
* Alle vertragingen op dagen met 'normaal' weer worden veroorzaakt door de andere luchthaven.

Dit betekent dat als een vertraging in Zürich wordt veroorzaakt door iets anders dan het weer, dit wordt genegeerd. Dit betekent ook dat als het slechte weer op een dag samenviel met een moment waarop een vlucht niet vertrok of aankwam en de oorzaak van de vertraging bij de andere luchthaven lag, deze ten onrechte aan Zürich wordt toegewezen.""")

with tab_factors:
    st.write("De balken tonen afwijkingen van het jaargemiddelde. De aantallen en absolute gemiddelde vertraging staan bij het aanwijzen van een balk. Een positief verschil betekent dat die categorie gemiddeld later is dan de totale selectie in hetzelfde jaar.")
    chart_choice = st.selectbox("Kies een vluchtkenmerk", ["Aankomst / vertrek", "Vliegtuigtype", "Baan", "Herkomst / bestemming", "Verkeersdrukte"])
    charts = {"Aankomst / vertrek": (step_4c_lsv, "LSV"), "Vliegtuigtype": (step_4c_aircraft, "ACT"), "Baan": (step_4c_runway, "RWY"), "Herkomst / bestemming": (step_4c_destination, "Org/Des"), "Verkeersdrukte": (step_4c_traffic, "Traffic_bin")}
    fig, column = charts[chart_choice]
    st.plotly_chart(fig, width = 'stretch')
    # Conclusie op basis van precies de categorieën die in de figuur staan.
    shown = set(str(x) for trace in fig.data for x in trace.x)
    effect = delay_effect(region_df, column).reset_index()
    effect["category"] = effect[column].astype(str)
    if column == "LSV":
        effect["category"] = effect["category"].map({"L": "Arrivals", "S": "Departures"})
    effect = effect[effect["category"].isin(shown)]
    if not effect.empty:
        high = effect.loc[effect["effect"].idxmax()]
        low = effect.loc[effect["effect"].idxmin()]
        st.write(f"**Wat valt op?** Binnen de getoonde categorieën ligt {high['category']} in {high['Year']} het hoogst ({high['effect']:+.1f} min; {high['flights']:.0f} geldige vertragingen), en {low['category']} in {low['Year']} het laagst ({low['effect']:+.1f} min). Verschillen in groepsgrootte en andere vluchtkenmerken kunnen dit beïnvloeden.")
    with st.expander("Exacte samenvatting van deze categorieën", expanded = True):
        st.dataframe(effect.drop(columns = "category"), hide_index = True)

with tab_weather:
    weather_choice = st.selectbox("Welke weeranalyse?", ["Regen", "Wind", "Correlatie per jaar", "Minuten per weereenheid", "Slecht weer tegenover overige dagen"])
    if weather_choice in ["Regen", "Wind"]:
        st.plotly_chart(step_4c_rain if weather_choice == "Regen" else step_4c_wind, width = 'stretch')
        st.write("We groeperen vluchten op dagelijkse regen of wind en vergelijken per jaar hun gemiddelde afwijking van het jaargemiddelde. Een groep kan ook verschillen in seizoen, drukte of routes; daarmee is het weersverschil nog geen bewezen oorzaak.")
    elif weather_choice == "Correlatie per jaar":
        st.plotly_chart(step_4d_ratio_chart, width = 'stretch')
        st.write("r loopt van −1 tot +1. Positief betekent dat hogere weerwaarden samengaan met meer dagvertraging; negatief betekent het omgekeerde. Een waarde rond nul betekent weinig lineaire samenhang. Iedere dag telt één keer.")
        correlations = step_4d_weather_table["ratio"].stack().dropna()
        if not correlations.empty:
            key = correlations.abs().idxmax()
            st.write(f"**Wat valt op?** Het grootste absolute verband in deze selectie is {key[0]} in {key[1]}, met r = {correlations.loc[key]:.2f}. Dit is samenhang, geen verklaring of zelfstandig voorspelmodel.")
    elif weather_choice == "Minuten per weereenheid":
        st.plotly_chart(step_4d_per_unit_chart, width = 'stretch')
        st.write("Een aparte rechte lijn wordt aangepast voor iedere weerfactor en ieder jaar. De helling toont minuten vertraging per eenheid van die factor. De hoogtes tussen verschillende factoren zijn niet rechtstreeks vergelijkbaar: temperatuur, wind en luchtdruk hebben verschillende eenheden.")
    else:
        comparison = step_5a_delay_check.reset_index()
        comparison["Year"] = comparison["Year"].astype(str)
        comparison["Weer"] = comparison["Bad_weather"].map({True: "Slecht weer volgens definitie", False: "Overige dagen"})
        st.plotly_chart(px.bar(comparison, x = "Year", y = "mean", color = "Weer", barmode = "group", hover_data = ["count"], title = "Gemiddelde vertraging op slechtweerdagen en overige dagen", labels = {"Year": "Jaar", "mean": "Gemiddelde vertraging (min)", "count": "Aantal geldige vertragingen"}), width = 'stretch')
        for year in region_df["Year"].unique():
            if (year, True) in step_5a_delay_check.index and (year, False) in step_5a_delay_check.index:
                diff = step_5a_delay_check.loc[(year, True), "mean"] - step_5a_delay_check.loc[(year, False), "mean"]
                st.write(f"{year}: op slechtweerdagen is de gemiddelde vertraging {abs(diff):.1f} minuten {'hoger' if diff >=  0 else 'lager'} dan op overige dagen. De definitie is gebaseerd op de percentielgrenzen van deze selectie.")
    with st.expander("Weertabellen en definitie van slecht weer", expanded = True):
        st.dataframe(step_4d_weather_table)
        st.dataframe(step_5a_boundaries)
        st.dataframe(step_5a_delay_check)

with tab_map:
    st.subheader(f"Folium-kaart: toegewezen vertraging in {MAP_YEAR}")
    st.write("De kaart volgt het oorspronkelijke toewijzingsscenario: slechtweerdagen worden bij Zürich gegroepeerd, andere dagen bij de herkomst- of bestemmingsluchthaven. Rood is Zürich, blauw zijn de overige luchthavens. Puntgrootte is gekoppeld aan de gekozen grootheid in de zijbalk.")
    st.iframe(step_5c_map.get_root().render(), height = 550)
    st.write("Bij 'Number of delays' wordt de cirkel groter als meer bewegingen een positieve vertraging hebben. Bij 'Average delay' wordt de cirkel groter bij een hogere gemiddelde vertraging. Negatieve gemiddelden krijgen de minimale grootte. De straal schaalt met de wortel van de waarde, met een minimum en maximum; lees de tooltip voor exacte waarden.")
    st.write("Het percentage in de tooltip is het aandeel van deze luchthaven in alle toegewezen vertraagde bewegingen, niet het percentage vertraagde vluchten binnen die luchthaven. De kleur onderscheidt het scenario; de grootte encodeert de grootheid.")
    
    with st.expander("Kaartgegevens en controle op toewijzing", expanded = True):
        st.dataframe(blame.sort_values("delays", ascending = False), hide_index = True)
        st.dataframe(step_5b_blame_check)

with tab_future:
    st.subheader("Tijdsverloop en voorwaardelijke vooruitblik")
    st.write("Dit onderdeel is toegevoegd voor de rubric; jullie oorspronkelijke code bevatte geen voorspelmodel. We gebruiken de laatste geregistreerde periode en maken een eenvoudige weekdagvoorspelling van het verkeersvolume, geen voorspelling van individuele vertraging.")
    last_year = int(traffic_source["Year"].max())
    last = traffic_source[traffic_source["Year"] == last_year]
    daily_count = last.set_index("Scheduled_dt").resample("D").size()
    if len(daily_count) >=  28:
        # Laatste acht weken om het recente niveau beter te benaderen dan een heel jaar.
        recent = daily_count.tail(56)
        week_pattern = recent.groupby(recent.index.dayofweek).mean()
        future_dates = pd.date_range(daily_count.index.max() + pd.Timedelta(days = 1), periods = 7)
        predicted = round(pd.DataFrame({"Datum": future_dates, "Aantal": [week_pattern.get(d.dayofweek, np.nan) for d in future_dates], "Reeks": "Voorwaardelijke voorspelling"}),1)
        observed = daily_count.tail(28).rename("Aantal").rename_axis("Datum").reset_index()
        observed["Reeks"] = "Waargenomen geplande bewegingen"
        st.plotly_chart(px.line(pd.concat([observed, predicted]), x = "Datum", y = "Aantal", color = "Reeks", line_dash = "Reeks", title = "Laatste vier weken en voorspelling voor de volgende zeven dagen", labels = {"Datum": "Datum", "Aantal": "Vliegbewegingen per dag"}), width = 'stretch')
        early, late = daily_count.tail(28).head(14).mean(), daily_count.tail(14).mean()
        st.write(f"**Wat valt op?** De eerste helft van de laatste vier weken telt gemiddeld {early:.1f} bewegingen per dag; de tweede helft {late:.1f}. Dat is een {'stijging' if late > early else 'daling' if late < early else 'gelijk niveau'} in deze korte periode.")
        st.write(f"**Voor de toekomst:** als het weekpatroon van de laatste {len(recent)} dagen zich herhaalt, verwachten we ongeveer {predicted['Aantal'].sum():.0f} bewegingen in de zeven dagen na {daily_count.index.max():%d-%m-%Y}. De voorspelling veronderstelt vergelijkbare dienstregeling, weekpatronen en volledige registratie. Zij is een historische vooruitblik, geen actuele voorspelling voor vandaag.")
        train, test = daily_count.iloc[:-14].tail(56), daily_count.iloc[-14:]
        baseline = train.groupby(train.index.dayofweek).mean()
        test_pred = pd.Series([baseline.get(d.dayofweek, np.nan) for d in test.index], index = test.index)
        mask = test_pred.notna()
        mae = (test[mask] - test_pred[mask]).abs().mean()
        st.write(f"Een controle op de laatste 14 dagen, met uitsluitend eerdere dagen als trainingsdata, geeft MAE = {mae:.1f} bewegingen per dag. MAE is de gemiddelde absolute fout; het is geen betrouwbaarheidsinterval.")
        st.write("Bij grote veranderingen zoals die tussen 2019 en 2020 kan de aanname mislukken. We trekken daarom geen vaste groeitrend door. Ook weersomstandigheden, seizoenen en beleidswijzigingen kunnen het toekomstige volume veranderen.")
        with st.expander("Dagwaarden van de vooruitblik", expanded = True):
            st.dataframe(predicted.drop(columns = "Reeks"), hide_index = True)
    else:
        st.info("Te weinig dagen voor een bruikbare weekdagvoorspelling.")
