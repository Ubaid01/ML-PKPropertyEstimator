import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pakistan House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_features(X_df, is_train=True, train_stats=None):
    X = X_df.copy()
    X['room_density'] = (X['bedrooms'] + X['baths']) / X['area_sqft']
    X['bed_bath_ratio'] = X['bedrooms'] / (X['baths'] + 1)
    X['listing_age'] = 2025 - X['year_added'] # Use 2025 as in training.
    X['is_recent_list'] = (X['listing_age'] <= 2).astype(int)
    
    if is_train:
        area_bins = [0, X['area_sqft'].quantile(0.20), X['area_sqft'].quantile(0.40),
                     X['area_sqft'].quantile(0.60), X['area_sqft'].quantile(0.80), np.inf]
        bedroom_bins = [0, 2, 3, 5, np.inf]
        train_stats = {
            'area_bins': area_bins,
            'bedroom_bins': bedroom_bins
        }
    
    if train_stats:
        X['area_category'] = pd.cut(
            X['area_sqft'],
            bins=train_stats['area_bins'],
            labels=['Small', 'Medium', 'Large', 'VeryLarge', 'Luxury'],
            include_lowest=True
        )
        X['bedroom_category'] = pd.cut(
            X['bedrooms'],
            bins=train_stats['bedroom_bins'],
            labels=['1-2BR', '3BR', '4-5BR', '6+BR'],
            include_lowest=True
        )
    else:
        X['area_category'] = 'Medium'
        X['bedroom_category'] = '3BR'
    
    return X, train_stats

@st.cache_resource
def load_models():
    try:
        model_sale = joblib.load("model_sale.pkl")
        model_rent = joblib.load("model_rent.pkl")
        return model_sale, model_rent, True
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None, False

@st.cache_data
def load_data_and_stats():
    try:
        df = pd.read_csv("cleaned_data.csv")
        stats_sale = joblib.load("train_stats_sale.pkl")
        stats_rent = joblib.load("train_stats_rent.pkl")
        return df, stats_sale, stats_rent, True
    except FileNotFoundError as e:
        st.error(f"❌ Data files not found: {e}")
        return pd.DataFrame(), None, None, False

model_sale, model_rent, models_loaded = load_models()
df, stats_sale, stats_rent, data_loaded = load_data_and_stats()

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    } 
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 Pakistan House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get instant price estimates for properties across major Pakistani cities</div>', unsafe_allow_html=True)

if not models_loaded or not data_loaded:
    st.error("⚠️ Cannot start app: Missing required files (models or data).")
    st.stop()

with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app predicts property prices in Pakistan using ML-models trained on real estate data from major cities.
    
    **Cities Covered:**
    - Karachi
    - Lahore
    - Islamabad
    - Rawalpindi
    - Faisalabad
    """)
    
    st.divider()
    
    if not df.empty:
        st.header("📈 Dataset Stats")
        st.metric("Total Properties", f"{len(df):,}")
        st.metric("For Sale", f"{len(df[df['purpose']=='For Sale']):,}")
        st.metric("For Rent", f"{len(df[df['purpose']=='For Rent']):,}")
        
        st.divider()
        
        st.subheader("💰 Price Ranges")
        sale_median = df[df['purpose']=='For Sale']['price'].median()
        rent_median = df[df['purpose']=='For Rent']['price'].median()
        st.write(f"**Sale Median:** PKR {sale_median:,.0f}")
        st.write(f"**Rent Median:** PKR {rent_median:,.0f}")

st.header("Property Details")
province_city = {
    "Punjab": ["Lahore", "Faisalabad", "Rawalpindi"],
    "Sindh": ["Karachi"],
    "Islamabad Capital Territory": ["Islamabad"]
}
prop_types = sorted( df['property_type'].unique() ) if not df.empty else [
    'House', 'Flat', 'Upper Portion', 'Lower Portion', 'Farm House', 'Room'
]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location")
    purpose = st.selectbox(
        "Purpose",
        ['For Sale', 'For Rent'],
        help="Are you looking to buy or rent?"
    )
    province = st.selectbox(
        "Province",
        list(province_city.keys()),
        help="Select the province"
    )
    city = st.selectbox(
        "City",
        province_city[province],
        help="Select the city within the province"
    )

with col2:
    st.subheader("Property Type")
    property_type = st.selectbox(
        "Type",
        prop_types,
        help="Type of property"
    )
    year_added = st.number_input(
        "Year Listed",
        min_value=2018,
        max_value=2019,
        value=2019,
        help="How old is the property listed?"
    )

with col3:
    st.subheader("Specifications")
    area_sqft = st.number_input(
        "Area (sqft)",
        min_value=250,
        max_value=23900,
        value=1100,
        step=100,
        help="Total area in square feet"
    )
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        help="Number of bedrooms"
    )
    baths = st.number_input(
        "Bathrooms",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        help="Number of bathrooms"
    )

st.markdown("<br>", unsafe_allow_html=True)
submitted = st.button("🔮 Predict Price", use_container_width=True, type="primary")

if submitted:
    with st.spinner("🔄 Calculating price prediction..."):
        try:
            if purpose == 'For Sale':
                model = model_sale
                stats = stats_sale
            else:
                model = model_rent
                stats = stats_rent
            
            input_data = pd.DataFrame({
                'area_sqft': [area_sqft],
                'bedrooms': [bedrooms],
                'baths': [baths],
                'city': [city],
                'property_type': [property_type],
                'province_name': [province] ,
                'year_added': [year_added]
            })
            
            input_data_fe, _ = add_features(input_data, is_train=False, train_stats=stats)            
            input_data_fe = input_data_fe.drop(['year_added'], axis=1, errors='ignore')
            
            try:
                expected_columns = model.feature_names_in_
                input_data_fe = input_data_fe[expected_columns]
            except AttributeError:
                pass
            
            prediction_log = model.predict(input_data_fe)
            predicted_price = np.expm1(prediction_log[0])
            st.markdown(
                f'<div class="prediction-box">💰 Estimated {purpose} Price:<br>PKR {predicted_price:,.0f}</div>',
                unsafe_allow_html=True
            )
            st.balloons()            
            
            col1, col2, col3 = st.columns(3)
            with col1:
                price_per_sqft = predicted_price / area_sqft
                st.metric(
                    "Price per Sqft",
                    f"PKR {price_per_sqft:,.0f}",
                    help="Estimated price per square foot"
                )
            
            with col2:
                area_marla = area_sqft / 272.25
                st.metric(
                    "Area (Marla)",
                    f"{area_marla:.1f}",
                    help="Property area in marlas"
                )
            
            with col3:
                room_density = (bedrooms + baths) / area_sqft
                density_label = "Spacious" if room_density < 0.002 else "Compact" if room_density > 0.004 else "Standard"
                st.metric(
                    "Space Rating",
                    density_label,
                    help="Based on room density"
                )
            
            st.divider()
            st.header("🔍 Similar Properties in Market")
            
            with st.spinner("Finding similar properties..."):
                city_type_match = df[
                    (df["purpose"] == purpose) &
                    (df["city"] == city) &
                    (df["property_type"] == property_type)
                ].copy()

                # 1: Exact Match
                exact_match = city_type_match[
                    (city_type_match["bedrooms"] == bedrooms) &
                    (city_type_match["baths"] == baths)
                ].copy()
                exact_match['match_type'] = 'Exact Match'
                
                # 2. City Relaxed
                province_match = df[
                    (df["purpose"] == purpose) &
                    (df["province_name"] == province) &
                    (df["property_type"] == property_type) &
                    (df["city"] != city)
                ].copy()
                
                # 3. Type Relaxed
                city_match = df[
                    (df["purpose"] == purpose) &
                    (df["city"] == city) &
                    (df["property_type"] != property_type)
                ].copy()
                
                # 4. Rooms Relaxed
                rooms_match = city_type_match[
                    (city_type_match["bedrooms"] != bedrooms) |
                    (city_type_match["baths"] != baths)
                ].copy()

                exact_match['match_type'] = 'Exact Match'
                province_match['match_type'] = 'Same Province'
                city_match['match_type'] = 'Same City'
                rooms_match['match_type'] = 'Diff. Rooms'
                all_similar = pd.concat( [exact_match, rooms_match, province_match, city_match], ignore_index=True )
                
                if not all_similar.empty:
                    all_similar['price_diff'] = abs(all_similar['price'] - predicted_price)
                    all_similar['price_diff_pct'] = (all_similar['price_diff'] / predicted_price) * 100
                    
                    similar_top = all_similar.sort_values('price_diff').head(5)
                    st.success(f"✅ Found {len(all_similar):,} similar properties")
            
                    display_df = similar_top[[
                        'match_type', 'price', 'city', 'property_type', 
                        'area_sqft', 'bedrooms', 'baths', 'price_diff_pct'
                    ]].copy()
                    display_df['price'] = display_df['price'].apply(lambda x: f"PKR {x:,.0f}")
                    display_df['area_sqft'] = display_df['area_sqft'].apply(lambda x: f"{x:,.0f} sqft")
                    display_df['price_diff_pct'] = display_df['price_diff_pct'].apply(lambda x: f"{x:.1f}%")
                    display_df = display_df.rename(columns={
                        'match_type': 'Match Type',
                        'price': 'Price',
                        'city': 'City',
                        'property_type': 'Type',
                        'area_sqft': 'Area',
                        'bedrooms': 'Beds',
                        'baths': 'Baths',
                        'price_diff_pct': 'Price Difference'
                    })
                    
                    st.dataframe(
                        display_df,
                        hide_index=True,
                        width='stretch'
                    )
                    st.subheader("⚖️ Price Distribution of Similar Properties")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(similar_top['price']/1e6, bins=20, edgecolor='black', alpha=0.7, color='#1E88E5')
                    ax.axvline(predicted_price/1e6, color='red', linestyle='--', linewidth=2, label='Your Prediction')
                    ax.set_xlabel('Price (Millions PKR)', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.set_title('Price Distribution of Similar Properties', fontsize=14)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)        
                else:
                    st.info("ℹ️ No similar properties found in the dataset.")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏠 <b>Pakistan House Price Predictor</b> | Built with Streamlit & Machine Learning</p>
    <p style='font-size: 0.9rem;'>Data sourced from zameen.com | Models trained on 100K+ properties</p>
</div>
""", unsafe_allow_html=True)