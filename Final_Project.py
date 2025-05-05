"""
NAME: Matthew Guida
CS230: Section4
Data: New York Housing Market
URL:

Description:
This interactive dashboard analyzes and visualizes New York City housing market data, providing users with comprehensive
insights into property listings across different price ranges and neighborhoods. The application features an interactive
map showing property locations, price distribution charts, and detailed filters that allow users to explore homes by bedrooms,
bathrooms, price range, and location. It presents key market statistics, categorizes properties into basic, average and
luxury segments, and enables users to identify trends through multiple visualization formats including heatmaps, cluster
maps, and comparative charts. The output serves as a powerful real estate exploration tool that helps potential buyers
understand market conditions and property distributions across NYC.
"""

import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster , HeatMap

data = pd.read_csv(r"C:\Users\mjg04\OneDrive - Bentley University\Spring Semester 2025\CS 230\Final Project\NY-House-Dataset(in).csv")
#Just putting the file name itself was creating an issue, So i found if i linked to the exact location it resolved the problem


#[DA1] Cleaning the code
data = data.dropna()
data["PRICE"] = data["PRICE"].astype(int)
data["BEDS"] = data["BEDS"].astype(int)
data["BATH"] = data["BATH"].astype(int)
data["PROPERTYSQFT"] = data["PROPERTYSQFT"].astype(int)
data["LATITUDE"] = data["LATITUDE"].astype(float)
data["LONGITUDE"] = data["LONGITUDE"].astype(float)
data["BATH"] = data["BATH"].round(0)
data["PROPERTYSQFT"] = data["PROPERTYSQFT"].round(0)
#get rid of duplicates, keeping the first one
data = data.drop_duplicates(subset=["FORMATTED_ADDRESS"])
#All the data points are from NY, and the STATE column is actually the city name where ever home is, so the code below changes the column name to CITY from STATE
data.rename(columns = {"STATE": "CITY"}, inplace=True)
#The code below does the same but for the SUBLOCALITY column because it is actually describing counties where the homes are
data.rename(columns = {"SUBLOCALITY": "COUNTY"}, inplace=True)

data["COUNTY "] = data["COUNTY"]
data = data.set_index("COUNTY ")

#[DA2] Sort data in ascending or descending order, by one or more columns
sorted_data = data.sort_values(by="PRICE", ascending=False)
data = sorted_data
#[DA9] Add a new column or perform calculations on DataFrame columns
data["PRICE_PERSQFT"] = data["PRICE"] / data["PROPERTYSQFT"]
data["PRICE_PERSQFT"] = data["PRICE_PERSQFT"].round(2)
data["PRICE_PERSQFT"] = "$" + data["PRICE_PERSQFT"].astype(str)

#[DA7] Add/drop/select/create new/group columns
data.drop(columns=["LONG_NAME", "LOCALITY", "ADDRESS", "MAIN_ADDRESS", "ADMINISTRATIVE_AREA_LEVEL_2"], inplace=True)
#all 5 columns dropped are redundant and are other columns so this helps clean the data

#[DA3] Find Top largest or smallest values of a column
overall_min_bath = min(data["BATH"])
overall_max_bath = max(data["BATH"])
overall_min_beds = min(data["BEDS"])
overall_max_beds = max(data["BEDS"])


#[ST4]
# Add custom CSS at the top
#Changes the background color of the website
st.markdown("""
<style>
    /* Main background - dark blue */
    .stApp {
        background-color: #1a3e72;  /* Rich dark blue */
        background-image: none;
    }

    /* Content container - white with enhanced styling */
    .main .block-container {
        background-color: white;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 1.5rem;
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* Improve header visibility */
    header {
        background-color: white !important;
        border-bottom: 1px solid #e1e1e1 !important;
    }
</style>
""", unsafe_allow_html=True)

#changes the font of the words
st.markdown("""
<style>
    /* Change font for all text */
    html, body, [class*="css"]  {
        font-family: 'Arial', sans-serif;
    }

    /* Change title color and font */
    h1 {
        color: #4a4a4a;
        font-size: 2.5rem !important;
    }

    /* Change sidebar color */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }

    /* Change button color */
    .stButton>button {
        background-color: #4a8fe7;
        color: white;
    }

    /* Change hover color */
    .stButton>button:hover {
        background-color: #3a7bd5;
        color: white;
    }
</style>
""", unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Main content area */
    .main .block-container {
        background-color: #ffffff;  /* White */
        padding: 2rem;
        border-radius: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #4b6cb7, #182848);  /* Blue gradient */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>New York City Housing Market</h1>", unsafe_allow_html=True)
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu_%28cropped%29.jpg/1200px-View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu_%28cropped%29.jpg",
    caption="New York City Skyline",  use_container_width=True)

#[PY1] Enhanced filtering function
def filter_housing_data(data, beds=None, baths=None, price_range=(None, None)):
    """
    Filters properties with multiple criteria
    :param data: Property DataFrame
    :param beds: Tuple of (min_beds, max_beds) or None
    :param baths: Minimum bathrooms or None
    :param price_range: Tuple of (min_price, max_price)
    :return: Filtered DataFrame
    """
    filtered = data.copy()

    if beds:
        filtered = filtered[(filtered['BEDS'] >= beds[0]) & (filtered['BEDS'] <= beds[1])]

    if baths:
        filtered = filtered[filtered['BATH'] >= baths]

    if price_range[0]:
        filtered = filtered[filtered['PRICE'] >= price_range[0]]

    if price_range[1]:
        filtered = filtered[filtered['PRICE'] <= price_range[1]]

    return filtered, len(filtered)

basic_homes_df, basic_count = filter_housing_data(
    data,
    beds=(1, 3),
    baths=2,
    price_range=(None, 1499999)
)

luxury_homes_df, luxury_count = filter_housing_data(
    data,
    beds=(4, overall_max_beds),
    baths=3,
    price_range=(1500000, None)
)

# Update counts after additional filtering
total_homes = len(data)
average_homes = total_homes - basic_count - luxury_count

#Display the count of basic and luxury homes
st.subheader("Types of homes break down in NYC")
st.write(f"Total number of homes in our data is: {total_homes}")
st.write(f"A Basic home has no more than 3 bedrooms, no more than 2 bathrooms, and is less than $1,500,000. Number of Basic homes: {basic_count}")
st.write(f"A Luxury home has more than 3 bedrooms, more than 2 bathroom, and is at least $1,500,000. Number of Premium homes: {luxury_count}")
st.write(f"An Average home is one that doesn't fit into one of the two categories above. Number of Average homes: {average_homes}")

st.subheader("What type of home are you looking for?")
#[ST1]
beds = st.radio("Pick how many bedrooms you want.", ["1-2", "3-4", "5-6", "7-9", "10+"])
if beds == "1-2":
    min_beds, max_beds = 1, 2
elif beds == "3-4":
    min_beds, max_beds = 3, 4
elif beds == "5-6":
    min_beds, max_beds = 5, 6
elif beds == "7-9":
    min_beds, max_beds = 7, 9
else:
    min_beds, max_beds = 10, overall_max_beds

#[ST2]
baths = st.slider("How many Baths do you want?", overall_min_bath,overall_max_bath)
#[ST3]
price = st.text_input("Enter your budget (as a whole number): ")

#[PY4] A list comprehension
# Get unique counties from the data
counties = data.index.unique().tolist()
# [ST3] Add a dropdown menu for county selection
selected_county = st.selectbox("Select a county:", ["All"] + counties)
listing_type = data["TYPE"].unique().tolist()
selected_listing = st.selectbox("Select listing type:", ["All"] + listing_type)

#[PY3] Error checking with try/except default pt
try:
    if not price.isdigit():
        # If the input is not all digits, raise an error to be caught by the except block.
        raise ValueError("The budget must be a number.")
    budget = int(price)
except ValueError:
    st.error("Please enter a valid number for your budget.")
    budget = 1200000

#[DA5] Filter data by two or more conditions with AND or OR
# Filter data based on the selected number of bedrooms, baths, and budget.
user_filtered_data = data[
    (data["BEDS"] >= min_beds) & (data["BEDS"] <= max_beds) &
    (data["BATH"] >= baths) &
    (data["PRICE"] <= budget)
]

# Apply county and listing type filters if selected
if selected_county != "All":
    basic_homes_df = basic_homes_df[basic_homes_df.index == selected_county]
    luxury_homes_df = luxury_homes_df[luxury_homes_df.index == selected_county]

if selected_listing != "All":
    basic_homes_df = basic_homes_df[basic_homes_df["TYPE"] == selected_listing]
    luxury_homes_df = luxury_homes_df[luxury_homes_df["TYPE"] == selected_listing]

#[PY2] Function that returns multiple values
#Returns: (avg_price, median_price, min_price, max_price)
def calculate_price_stats(df):
    avg = df['PRICE'].mean()
    median = df['PRICE'].median()
    min_p = df['PRICE'].min()
    max_p = df['PRICE'].max()
    return avg, median, min_p, max_p

# Calculate statistics for filtered data
avg_price, median_price, min_price, max_price = calculate_price_stats(user_filtered_data)

# Display in Streamlit
st.subheader("Price Statistics for Filtered Properties")
col1, col2 = st.columns(2)
with col1:
    st.metric("Average Price", f"${avg_price:,.0f}")
    st.metric("Minimum Price", f"${min_price:,.0f}")
with col2:

    st.metric("Maximum Price", f"${max_price:,.0f}")
    st.metric("Median Price", f"${median_price:,.0f}")

# Display the filtered property listings.
st.subheader("Filtered Property Listings")
#Displays all the filtered listings
st.write(f"Showing {len(user_filtered_data)} properties based on your filters:")
st.dataframe(user_filtered_data)

#[MAP]
# Display an Interactive Map of Property Locations
st.subheader("Map of Property Locations")
# For mapping, reset the index so COUNTY becomes a column again.
map_df = user_filtered_data.reset_index()

if not map_df.empty:
    # Ensure we have valid coordinates
    map_df = map_df.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # Create PyDeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=user_filtered_data,
        get_position=["LONGITUDE", "LATITUDE"],
        get_radius=100,
        get_fill_color="color",
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        auto_highlight=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1
    )

    # Set view state with precise coordinates
    view_state = pdk.ViewState(
        latitude=user_filtered_data["LATITUDE"].mean(),
        longitude=user_filtered_data["LONGITUDE"].mean(),
        zoom=9,
        pitch=40
    )

    # Enhanced tooltip configuration
    #this shows the information for each house when you hover over it on the map
    tooltip = {
        "html": """
                <div style="
                    background-color: #2a3f5f;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.2);
                ">
                    <h3 style="margin-top: 0; margin-bottom: 5px;">Property Details</h3>
                    <p style="margin: 5px 0;"><b>Price:</b> ${PRICE}</p>
                    <p style="margin: 5px 0;"><b>Beds:</b> {BEDS} | <b>Baths:</b> {BATH}</p>
                    <p style="margin: 5px 0;"><b>Size:</b> {PROPERTYSQFT} sqft</p>
                    <p style="margin: 5px 0;"><b>Price/sqft:</b> {PRICE_PERSQFT}</p>
                    <p style="margin: 5px 0;"><b>Location:</b> {CITY}</p>
                </div>
            """,
        "style": {
            "backgroundColor": "#2a3f5f",
            "color": "white"
        }
    }

    # Map interaction controls
    with st.expander("Map Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            zoom_level = st.slider("Zoom Level", 5, 15, 9)
        with col2:
            pitch_level = st.slider("Map Tilt", 0, 60, 40)

        # Update view state with user controls
        view_state.zoom = zoom_level
        view_state.pitch = pitch_level

    # Render map
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v10"
    ))
#[FOLIUM1]
    # 2. Folium Cluster Map
    st.header("📍 Property Cluster Map")
    if not user_filtered_data.empty:
        m = folium.Map(
            location=[user_filtered_data["LATITUDE"].mean(), user_filtered_data["LONGITUDE"].mean()],
            zoom_start=10
        )

        marker_cluster = MarkerCluster().add_to(m)

        for _, row in user_filtered_data.iterrows():
            folium.Marker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                popup=f"""
                    <b>${row['PRICE']:,}</b><br/>
                    {row['BEDS']} beds, {row['BATH']} baths<br/>
                    {row['PROPERTYSQFT']:,} sqft<br/>
                    {row['TYPE']}<br/>
                    {row['CITY']}, {row['COUNTY']}
                """,
                icon=folium.Icon(color="green")
            ).add_to(marker_cluster)

        folium_static(m)
#[SEA1]
    #Seaborn Price Distribution
    st.header("📊 Price Distribution Analysis")
    st.markdown("Visualizing price trends across different property types")

    col1, col2 = st.columns(2)
    with col1:
        # Seaborn Plot 1
        st.subheader("Price by Bedroom Count")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=user_filtered_data,
            x="BEDS",
            y="PRICE",
            palette="YlGn",
            ax=ax1
        )
        ax1.set_yscale('log')
        ax1.set_title("Price Distribution by Bedrooms", fontsize=14)
        ax1.set_xlabel("Number of Bedrooms")
        ax1.set_ylabel("Price ($) - Log Scale")
        st.pyplot(fig1)
#[SEA2]
    with col2:
        # Seaborn Plot 2
        st.subheader("Price vs. Size")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=user_filtered_data,
            x="PROPERTYSQFT",
            y="PRICE",
            hue="BEDS",
            size="BATH",
            palette="viridis",
            sizes=(20, 200),
            alpha=0.7,
            ax=ax2
        )
        ax2.set_title("Price vs. Square Footage", fontsize=14)
        ax2.set_xlabel("Square Footage")
        ax2.set_ylabel("Price ($)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig2)

    #[CHART1]
    st.header("📋 Top Priced Properties")
    st.markdown("Table and chart showing the highest priced properties")

    tab1, tab2 = st.tabs(["Data Table", "Bar Chart"])

    with tab1:
        # Matplotlib Table
        st.dataframe(
            user_filtered_data.nlargest(10, "PRICE")[["PRICE", "BEDS", "BATH", "PROPERTYSQFT", "CITY", "COUNTY"]]
            .style.format({
                "PRICE": "${:,.0f}",
                "PROPERTYSQFT": "{:,.0f} sqft"
            })
        )
    #[CHART2]
    with tab2:
        # Matplotlib Bar Chart
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        top_properties = user_filtered_data.nlargest(10, "PRICE")
        bars = ax3.barh(
            top_properties["CITY"] + ", " + top_properties["COUNTY"],
            top_properties["PRICE"],
            color=sns.color_palette("YlOrRd", len(top_properties))
        )
        ax3.bar_label(bars, fmt='${:,.0f}', padding=5)
        ax3.set_title("Top 10 Most Expensive Properties", fontsize=14)
        ax3.set_xlabel("Price ($)")
        ax3.set_ylabel("Location")
        plt.tight_layout()
        st.pyplot(fig3)

    #[FOLIUM2]
    st.header("🔥 Price Density Heatmap")
    st.markdown("Visualizing price density across New York")

    if not user_filtered_data.empty:
        heat_map = folium.Map(
            location=[user_filtered_data["LATITUDE"].mean(), user_filtered_data["LONGITUDE"].mean()],
            zoom_start=10
        )

        from folium.plugins import HeatMap

        heat_data = user_filtered_data[["LATITUDE", "LONGITUDE", "PRICE"]].values.tolist()
        HeatMap(heat_data, radius=15).add_to(heat_map)

        folium_static(heat_map)

