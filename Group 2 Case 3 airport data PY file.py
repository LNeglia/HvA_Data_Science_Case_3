#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
import folium
import streamlit as st
import streamlit.components.v1 as components

# In[ ]:


# Step 1: Load/Cache datasets
@st.cache_data
def load_data():
    schedule_airport = pd.read_csv("schedule_airport.csv", na_values = "-")
    airports = pd.read_csv(
        "airports-extended.csv",
        names = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude",
               "Altitude", "Timezone", "DST", "Tz database", "Type", "Source"],
        na_values = "\\N",
    )
    zurich_weather = pd.read_csv(
        "06670.csv",
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


schedule_airport.head()


# In[ ]:


airports.head()


# In[ ]:


zurich_weather.head()


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

na_share = schedule_airport[weather_cols].isna().mean()
drop_cols = na_share[na_share > 0.5].index.tolist()
step_2d_dropped_columns = pd.DataFrame({"dropped_column": drop_cols, "missing_pct": (na_share[drop_cols] * 100).round(1).values})

weather_cols = [c for c in weather_cols if c not in drop_cols]
schedule_clean = schedule_airport.drop(columns = drop_cols).dropna(subset = weather_cols)

# Effect of the drop
before = schedule_airport.groupby("Year")["Delay"].agg(["count", "mean", "median"])
after  = schedule_clean.groupby("Year")["Delay"].agg(["count", "mean", "median"])
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

airports = airports.dropna(subset=["ICAO"]).drop_duplicates(subset="ICAO")
step_3a_airports_clean = airports.head()


# In[ ]:


# Step 3b: Merge on ICAO, count and drop the unmatched flights

schedule_clean = schedule_clean.merge(airports[["ICAO", "Name", "Country", "Latitude", "Longitude", "Tz database"]],
                                      left_on="Org/Des", right_on="ICAO",
                                      how="left", validate="many_to_one")

unmatched = schedule_clean["ICAO"].isna()
step_3b_unmatched_count = pd.DataFrame({"flights": [unmatched.sum()], "pct": [round(unmatched.mean() * 100, 2)]},
                                       index = ["Without airport match"])
step_3b_unmatched_codes = schedule_clean.loc[unmatched, "Org/Des"].value_counts().head(10).to_frame("flights")

before = schedule_clean.groupby("Year")["Delay"].agg(["count", "mean", "median"])
schedule_clean = schedule_clean.dropna(subset=["ICAO"])
after = schedule_clean.groupby("Year")["Delay"].agg(["count", "mean", "median"])
step_3b_before_after = pd.concat([before, after], axis=1, keys=["before", "after"]).round(1)


# In[ ]:


# Step 4a: Europe/Intercontinental using time zone and region selection

# European countries that don't have "Europe/" as their time zone entry 
extra_europe = ["Atlantic/Canary", "Atlantic/Azores", "Atlantic/Reykjavik"]

is_europe = (schedule_clean["Tz database"].str.startswith("Europe/", na=False)
             | schedule_clean["Tz database"].isin(extra_europe))
schedule_clean["Region"] = np.where(is_europe, "Europe", "Intercontinental")

step_4a_region_counts = schedule_clean.groupby(["Year", "Region"]).size().unstack()
step_4a_region_counts["No time zone"] = (schedule_clean[schedule_clean["Tz database"].isna()]
                                         .groupby("Year").size().reindex(step_4a_region_counts.index, fill_value = 0))

# Region selection: one region or both. sorted + tuple, so the click order doesn't create a second cache entry
REGIONS = tuple(sorted(st.sidebar.multiselect("Region", ["Europe", "Intercontinental"], default = ["Europe", "Intercontinental"])))
if not REGIONS:
    st.warning("Kies min een regio")
    st.stop()

@st.cache_data
def get_region(_df, regions):
    return _df[_df["Region"].isin(regions)].copy()

region_df = get_region(schedule_clean, REGIONS)


# In[ ]:


# Functions to look at average delay per column, per year. Looks at average delay, then the average delay per category relative to overall.
# TLDR: Average delay per category, shows how far categorical average is from overall average.

def delay_effect(df, col):
    year_mean = df.groupby("Year")["Delay"].transform("mean")
    out = (df.assign(Effect=df["Delay"] - year_mean)
             .groupby(["Year", col], observed=True)
             .agg(flights=("Delay", "count"),
                  mean_delay=("Delay", "mean"),
                  effect=("Effect", "mean")))
    return out.round(1)

def compare_years(df, col):
    return (delay_effect(df, col)["effect"]
            .unstack("Year").dropna().sort_values(2019, ascending=False))


# In[ ]:


# Step 4b: Code for line chart for delay per hour/weekday/month and per year. Cached per region, period and years

PERIOD = st.sidebar.radio("Delay per", ["Hour", "Weekday", "Month"])
YEARS = tuple(sorted(st.sidebar.multiselect("Years in the line graph", [2019, 2020], default = [2019, 2020])))

@st.cache_data
def time_chart(_df, regions, period, years):
    df = _df[_df["Year"].isin(years)].copy()
    df["Effect"]  = df["Delay"] - df.groupby("Year")["Delay"].transform("mean")
    df["Weekday"] = df["STD"].dt.day_name()
    df["Month"]   = df["STD"].dt.month

    data = df.groupby(["Year", "LSV", period])["Effect"].mean().reset_index()
    data["Year"] = data["Year"].astype(str)
    return px.line(data, x=period, y="Effect", color="Year", line_dash="LSV",
                   category_orders={"Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                   title=f"{' + '.join(regions)}: minutes above/below yearly average, per {period.lower()}")

step_4b_time_chart = time_chart(region_df, REGIONS, PERIOD, YEARS)


# In[ ]:


# Step 4c: Effect per factor, as bar charts (cached per region)

@st.cache_data
def add_bins(_df, regions):
    df = _df.copy()
    # Traffic volume
    df["Traffic_bin"] = pd.cut(df["Traffic_hour"], bins=[0, 20, 40, 60, 80, 200])
    # Weather
    df["Rain_bin"] = pd.cut(df["Total Precipitation"], bins=[-0.1, 0, 1, 5, 100],
                            labels=["dry", "0-1 mm", "1-5 mm", ">5 mm"])
    # Extra code needed here: math with Floats was causing the lower bin to have 16 numbers after the decimal point
    wind_edges = df["Wind Speed"].quantile([0, 0.25, 0.5, 0.75, 1]).round(2).tolist()
    df["Wind_bin"] = pd.cut(df["Wind Speed"], bins=wind_edges, include_lowest=True,
                            labels=[f"{a}-{b}" for a, b in zip(wind_edges[:-1], wind_edges[1:])])
    return df

# If both years true, compares/shows only entries that share categories
@st.cache_data
def factor_chart(_df, regions, col, title, both_years=False, top=None):
    data = delay_effect(_df, col).reset_index()

    if both_years:
        order = compare_years(_df, col).head(top).index.tolist()
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

    fig = px.bar(data, x=col, y="effect", color="Year", barmode="group", hover_data=["flights", "mean_delay"],
                 category_orders={col: order, "Year": ["2019", "2020"]}, labels={"effect": "Minutes above/below yearly average"},
                 title=f"{' + '.join(regions)}: {title}")
    fig.update_traces(marker_line_color="black", marker_line_width=1)   # outline per bar
    return fig
# The original graph was set up by hand, but when changing to a function that takes both regions, AI was used to ensure there weren't any bugs


region_df = add_bins(region_df, REGIONS)

# LSV, plane type, runway, destination
step_4c_lsv         = factor_chart(region_df, REGIONS, "LSV", "arrivals vs. departures")
step_4c_aircraft    = factor_chart(region_df, REGIONS, "ACT", "aircraft type (top 15 in 2019)", both_years=True, top=15)
step_4c_runway      = factor_chart(region_df, REGIONS, "RWY", "runway", both_years=True)
step_4c_destination = factor_chart(region_df, REGIONS, "Org/Des", "origin/destination (top 15 in 2019)", both_years=True, top=15)
# Traffic volume
step_4c_traffic     = factor_chart(region_df, REGIONS, "Traffic_bin", "traffic volume (movements per hour)")
# Weather
step_4c_rain        = factor_chart(region_df, REGIONS, "Rain_bin", "daily precipitation")
step_4c_wind        = factor_chart(region_df, REGIONS, "Wind_bin", "daily wind speed (quartiles)")


# In[ ]:


# Step 4d: Weather ratio and minutes per unit, at day level (cached per region)

# @st.cache_data
def weather_ratios(_df, regions, weather_cols):
    # Weather slopes at day level
    daily = (_df.groupby(["Year", "STD"])
                .agg(mean_delay=("Delay", "mean"), **{column: (column, "first") for column in weather_cols}))
    rows = []
    for year, day in daily.groupby("Year"):
        for column in weather_cols:
            rows.append({"Year": str(year), "Weather": column,
                         "ratio": day[column].corr(day["mean_delay"]),
                         "min_per_unit": np.polyfit(day[column], day["mean_delay"], 1)[0]})
    data = pd.DataFrame(rows).round(2)

    label = " + ".join(regions)
    table = data.pivot(index="Weather", columns="Year", values=["ratio", "min_per_unit"])

    ratio_fig = px.bar(data, x="Weather", y="ratio", color="Year", barmode="group",
                       category_orders={"Year": ["2019", "2020"]},
                       labels={"ratio": "Ratio (correlation r) with daily mean delay"},
                       title=f"{label}: how strongly each weather value moves with the daily delay")
    unit_fig = px.bar(data, x="Weather", y="min_per_unit", color="Year", barmode="group",
                      category_orders={"Year": ["2019", "2020"]},
                      labels={"min_per_unit": "Minutes of delay per unit"},
                      title=f"{label}: minutes of delay per unit of each weather value")
    for fig in (ratio_fig, unit_fig):
        fig.update_traces(marker_line_color="black", marker_line_width=1)
    return table, ratio_fig, unit_fig

# The data was calculated without AI, but the conversion from a table to a bar chart to show the results used some AI for formatting and speed

step_4d_weather_table, step_4d_ratio_chart, step_4d_per_unit_chart = weather_ratios(region_df, REGIONS, tuple(weather_cols))


# In[ ]:


# Step 5a: Bad-weather boundaries and checks (cached per region)

@st.cache_data
def add_bad_weather(_df, regions):
    df = _df.copy()
    days = df.groupby("STD")[["Total Precipitation", "Wind Speed", "Air Pressure", "Temperature"]].first()

    rain_limit     = days["Total Precipitation"].quantile(0.9)   # top 10% wettest days
    wind_limit     = days["Wind Speed"].quantile(0.9)            # top 10% windiest days
    pressure_limit = days["Air Pressure"].quantile(0.1)          # bottom 10% lowest pressure
    temp_limit     = 0                                           # freezing point, °C

    df["Storm"]       = (df["Air Pressure"] < pressure_limit) & (df["Total Precipitation"] > rain_limit)
    df["Ice_risk"]    = (df["Temperature"] <= temp_limit) & (df["Total Precipitation"] > 0)
    df["Strong_wind"] = df["Wind Speed"] > wind_limit
    df["Bad_weather"] = df["Storm"] | df["Ice_risk"] | df["Strong_wind"]

    boundaries = pd.DataFrame({"Boundary": [f"> {rain_limit:.1f} mm", f"> {wind_limit:.1f} km/h",
                                            f"< {pressure_limit:.1f} hPa", f"<= {temp_limit} °C"]},
                              index=["Rain", "Wind", "Air Pressure", "Temperature"])
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

MAP_YEAR = st.sidebar.radio("Map year", [2019, 2020])

@st.cache_data
def blame_table(_df, _airports, regions, map_year):
    df = _df[_df["Year"] == map_year]
    blame = (df.assign(Blame_ICAO=np.where(df["Bad_weather"], "LSZH", df["ICAO"]),
                       Is_delayed=df["Delay"] > 0)
               .groupby("Blame_ICAO")
               .agg(flights=("Delay", "size"),
                    delays=("Is_delayed", "sum"),
                    mean_delay=("Delay", "mean"))
               .reset_index()
               .merge(_airports[["ICAO", "Name", "City", "Latitude", "Longitude"]],
                      left_on="Blame_ICAO", right_on="ICAO"))
    blame["share_pct"] = (blame["delays"] / blame["delays"].sum() * 100).round(1)

    check = pd.DataFrame({"value": [len(blame), blame["flights"].sum(), len(df)]},
                         index=["Airports on the map", "Flights credited", "Flights in selection"])
    return blame, check

blame, step_5b_blame_check = blame_table(region_df, airports, REGIONS, MAP_YEAR)


# In[ ]:


# Step 5c: Folium map. Changes based on chosen year and what the circles mean.

SIZE_BY = st.sidebar.radio("Circle size", ["delays", "mean_delay"], format_func={"delays": "Number of delays", "mean_delay": "Average delay"}.get)

def radius(value, highest, min_r=3, max_r=30):
    return min_r + (max_r - min_r) * np.sqrt(value / highest)

def build_map(blame, regions, size_by):
    blame = blame.assign(size=blame[size_by].clip(lower=0))
    highest = blame["size"].max()

    europe_only = regions == ("Europe",)
    flight_map = folium.Map(location=[47.46, 8.55] if europe_only else [20, 0],
                            zoom_start=4 if europe_only else 2)

    for _, row in blame.sort_values("size", ascending=False).iterrows():
        is_zrh = row["ICAO"] == "LSZH"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius(row["size"], highest),
            color="red" if is_zrh else "blue",
            fill=True,
            fill_opacity=0.7 if is_zrh else 0.5,
            weight=1,
            tooltip=f"{row['Name']} ({row['ICAO']})<br>Flights: {row['flights']}<br>"
                    f"Delayed: {row['delays']} ({row['share_pct']}%)<br>Avg delay: {row['mean_delay']:.1f} min",
        ).add_to(flight_map)
    return flight_map

step_5c_map = build_map(blame, REGIONS, SIZE_BY)


# In[ ]:


# In the empty cell below this one, put every step variable in order.


# In[ ]:


# Stap 1: de ruwe datasets
# .head() van het vliegschema, met "-" bij het inlezen als NaN gelezen
step_1a
# .head() van de lijst met luchthavenlocaties, met \N bij het inlezen als NaN gelezen
step_1b
# .head() van het dagelijkse weer in Zürich vanaf 1973
step_1c

# Stap 2: vertraging, samenvoegen met het weer en opschonen
# Dataframe na het toevoegen van de kolommen Delay (gecorrigeerd voor middernacht), Year, tijdstempels, Hour en Traffic_hour
step_2a_new_columns
# Dataframe na het samenvoegen met het weer in Zürich van 2019-2020 op datum
step_2b_weather_merged
# Percentage ontbrekende waarden per weerkolom
step_2c_weather_missing
# Aantal dagen in het schema tegenover het aantal dagen zonder weergegevens
step_2c_days_missing_weather
# Vluchten met minstens één ontbrekende weerwaarde, per jaar
step_2c_flights_missing_weather
# Weerkolommen die zijn verwijderd omdat ze voor meer dan 50% leeg zijn
step_2d_dropped_columns
# Aantal vluchten, gemiddelde en mediane vertraging per jaar, voor en na het verwijderen van rijen zonder weergegevens
step_2d_before_after

# Stap 3: luchthavenlijst en samenvoegen op ICAO
# Aantal ontbrekende waarden per kolom in de luchthavenlijst
step_3a_airport_na
# Dubbele ICAO-rijen, gesplitst in lege codes en echte codes die vaker voorkomen
step_3a_duplicates
# .head() van de luchthavenlijst na het verwijderen van lege en dubbele ICAO-codes
step_3a_airports_clean
# Aantal en percentage vluchten waarvan de Org/Des geen ICAO-match had
step_3b_unmatched_count
# De tien meest voorkomende Org/Des-codes zonder match
step_3b_unmatched_codes
# Aantal vluchten, gemiddelde en mediane vertraging per jaar, voor en na het verwijderen van vluchten zonder match
step_3b_before_after

# Stap 4: regio's en vertragingsfactoren
# Vluchten per jaar per regio, ingedeeld op de tijdzone van de luchthaven
step_4a_region_counts
# Minuten boven/onder het jaargemiddelde per uur, weekdag of maand, voor aankomsten en vertrekken
step_4b_time_chart
# Effect van aankomen of vertrekken op de vertraging
step_4c_lsv
# Effect van het vliegtuigtype, top 15 op basis van het effect in 2019
step_4c_aircraft
# Effect van de gebruikte baan
step_4c_runway
# Effect van de herkomst- of bestemmingsluchthaven, top 15 op basis van het effect in 2019
step_4c_destination
# Effect van de verkeersdrukte, ingedeeld op bewegingen per uur
step_4c_traffic
# Effect van de dagelijkse neerslag, ingedeeld van droog tot zware regen
step_4c_rain
# Effect van de dagelijkse windsnelheid, verdeeld in kwartielen
step_4c_wind
# Ratio en minuten vertraging per eenheid voor elke weerkolom, per jaar, op dagniveau
step_4d_weather_table
# Staafdiagram van de ratio tussen elke weerkolom en de gemiddelde dagvertraging
step_4d_ratio_chart
# Staafdiagram van de minuten vertraging per eenheid van elke weerkolom
step_4d_per_unit_chart

# Stap 5: slecht weer en de kaart
# Grenswaarden voor regen, wind, luchtdruk en temperatuur die slecht weer bepalen
step_5a_boundaries
# Aantal dagen per jaar met storm, ijzelrisico, harde wind of slecht weer in het algemeen
step_5a_condition_days
# Aantal vluchten en gemiddelde vertraging op dagen met slecht weer tegenover normale dagen, per jaar
step_5a_delay_check
# Controle dat elke vlucht in de selectie aan Zürich of aan de andere luchthaven is toegewezen
step_5b_blame_check
# Folium-kaart met de toegewezen vertragingen per luchthaven, met Zürich in rood voor vertragingen door slecht weer
components.html(step_5c_map._repr_html_(), height=500)

