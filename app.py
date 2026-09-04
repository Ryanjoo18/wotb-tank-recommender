import pandas as pd
import streamlit as st

from model import build_similarity_matrix, load_tank_data, recommend_tanks

st.set_page_config(
    page_title="WOTB Tank Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit caching
@st.cache_data(ttl=3600)
def get_tank_data():
    return load_tank_data()

@st.cache_resource
def get_similarity_matrix(tank_df):
    return build_similarity_matrix(tank_df)

def display_selected_tank(tank):
    """Displays the selected tank"""
    # create containers laid out as side-by-side columns
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"### {tank['name']}")
        st.image(tank["image_normal"])
            
    with right:
        st.metric(label="Tier", value=tank["tier"])
        st.metric(label="Type", value=tank["type"])
        st.metric(label="Nation", value=tank["nation"])
        st.metric(label="Status", value=("Premium" if tank["is_premium"] else "Tech tree"))

def display_recommendations(recommendations):
    """Displays recommendations of tanks similar to the selected tank"""
    for rank, (_, tank) in enumerate(recommendations.iterrows(), start=1):
        st.markdown(f"### {rank}. {tank['name']}")

        left, right = st.columns(2)
        with left:
            st.image(tank["image_preview"])

        with right:
            col1, col2 = st.columns([1,2])

            with col1:
                st.write(f"**Tier**: {int(tank['tier'])}")
                st.write(f"**Type**: {tank['type']}")
                st.write(f"**Nation**: {tank['nation']}")
                premium = ("Yes" if tank["is_premium"] else "No")
                st.write(f"**Premium**: {premium}")
                st.write(f"**Similarity**: {tank['similarity']:.3f}")

            with col2:
                st.write(f"**HP**: {tank['hp']:.0f}")
                st.write(f"**Average damage**: {tank['avg_damage']:.0f}")
                st.write(f"**Average penetration**: {tank['avg_penetration']:.0f}")
                st.write(f"**DPM**: {tank['dpm']:.0f}")
                st.write(f"**Top speed**: {tank['speed_forward']:.1f} km/h")

        st.divider()

def main():
    st.title("WOTB Tank Recommendation System")

    st.write("Select a tank to find other tanks with similar characteristics.")

    try:
        with st.spinner("Loading tank data..."):
            tank_df = get_tank_data()
            similarity_matrix = get_similarity_matrix(tank_df)

    except Exception as error:
        st.error("Unable to load the tank recommendation system.")
        st.exception(error)
        st.stop()

    # sidebar
    st.sidebar.header("Recommendation settings")
    tank_names = sorted(tank_df["name"].dropna().unique())
    selected_tank = st.sidebar.selectbox(label="Tank",
                                         options=tank_names)
    number_of_recommendations = st.sidebar.slider(
        label="Number of recommendations",
        min_value=1,
        max_value=20,
        value=10,
    )
    
    exclude_premium = st.sidebar.checkbox("Exclude premium tanks", value=False)

    # selected tank
    selected_tank_data = tank_df[tank_df["name"] == selected_tank].iloc[0]

    st.subheader("Selected Tank")
    display_selected_tank(selected_tank_data)

    st.divider()

    # recommendations
    matched_name, recommendations = recommend_tanks(tank_name=selected_tank,
                                                    tank_df=tank_df,
                                                    similarity_matrix=similarity_matrix,
                                                    number_of_recommendations=number_of_recommendations,
                                                    exclude_premium=exclude_premium
                                                    )

    if recommendations.empty:
        st.warning("No recommendations were found.")
        return

    st.subheader(f"Recommendations for {matched_name}")

    display_recommendations(recommendations)

if __name__ == "__main__":
    main()