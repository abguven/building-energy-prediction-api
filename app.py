import streamlit as st
import requests

# Configuration
# API_URL = "https://xxxxxxx.eu-west-3.awsapprunner.com/predict"
API_URL = "http://localhost:3000/predict"


 

st.set_page_config(
    page_title="Seattle Energy Predictor",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Building Energy Consumption Predictor — Seattle")
st.markdown("Estimate a building's annual energy consumption from its structural profile.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a Machine Learning model
    to predict energy consumption (in kBtu)
    for commercial buildings in Seattle.

    **Model**: Random Forest Regressor
    **API**: BentoML on AWS App Runner
    """)

# Form columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Building Information")

    property_use = st.text_input(
        "Primary property use type",
        value="Office",
        help="E.g.: Office, Retail, Warehouse"
    )

    all_uses = st.text_input(
        "All use types (comma-separated)",
        value="Office,Parking",
        help="E.g.: Office,Parking,Retail"
    )

    year_built = st.number_input(
        "Year built",
        min_value=1900,
        max_value=2025,
        value=2010,
        step=1
    )

    num_floors = st.number_input(
        "Number of floors",
        min_value=1,
        value=10,
        step=1
    )

with col2:
    st.subheader("🔢 Technical Characteristics")

    gfa_building = st.number_input(
        "Gross floor area — building (sq ft)",
        min_value=1,
        value=50000,
        step=1000
    )

    gfa_parking = st.number_input(
        "Gross floor area — parking (sq ft)",
        min_value=0,
        value=5000,
        step=500
    )

    energy_star_na = st.checkbox(
        "ENERGY STAR Score not available",
        help="Check if the score is unknown"
    )

    if not energy_star_na:
        energy_star = st.slider(
            "ENERGY STAR Score",
            min_value=1,
            max_value=100,
            value=75,
            help="Energy efficiency score (1-100)"
        )
    else:
        energy_star = None

st.markdown("---")

if st.button("🔮 Predict energy consumption", type="primary", use_container_width=True):

    payload = {
            "data":{
                "LargestPropertyUseType": property_use,
                "ListOfAllPropertyUseTypes": all_uses,
                "YearBuilt": year_built,
                "NumberofFloors": num_floors,
                "PropertyGFAParking": gfa_parking,
                "PropertyGFABuilding_s_": gfa_building,
                "ENERGYSTARScore": energy_star
            }
    }

    with st.spinner("Running prediction..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Prediction successful!")

                col_metric1, col_metric2 = st.columns(2)

                with col_metric1:
                    st.metric(
                        label="Predicted consumption",
                        value=f"{result['formatted']} kBtu"
                    )

                with col_metric2:
                    kwh = result['predicted_SiteEnergyUse(kBtu)'] * 0.293071
                    st.metric(
                        label="Equivalent",
                        value=f"{kwh:,.0f} kWh"
                    )

                if "warning_messages" in result and result["warning_messages"]:
                    st.warning("⚠️ **Warnings**")
                    for warning in result["warning_messages"]:
                        st.markdown(f"- {warning}")

                with st.expander("📊 Request details"):
                    st.json(payload)
                    st.json(result)

            else:
                st.error(f"❌ API error: {response.status_code}")
                st.json(response.json())

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built by Abdulkadir GUVEN | Model deployed on AWS App Runner
    </div>
    """,
    unsafe_allow_html=True
)