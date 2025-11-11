import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pakistan Property Price Predictor",
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

def find_similar_properties_by_priority(df, purpose, city, province, property_type, 
                                        bedrooms, baths, predicted_price, min_results=5):
    
    # 1. Exact Match
    exact_match = df[
        (df["purpose"] == purpose) &
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        (df["bedrooms"] == bedrooms) &
        (df["baths"] == baths)
    ].copy()
    
    if len(exact_match) >= min_results:
        exact_match['match_type'] = 'Exact Match'
        exact_match['price_diff'] = abs(exact_match['price'] - predicted_price)
        return exact_match.sort_values('price_diff'), 'Exact Match', len(exact_match)
    
    # 2. Relaxed Rooms
    rooms_relaxed = df[
        (df["purpose"] == purpose) &
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        ((df["bedrooms"] != bedrooms) | (df["baths"] != baths))
    ].copy()
    
    if len(rooms_relaxed) >= min_results:
        rooms_relaxed['match_type'] = 'Different Rooms'
        rooms_relaxed['price_diff'] = abs(rooms_relaxed['price'] - predicted_price)
        return rooms_relaxed.sort_values('price_diff'), 'Same City & Type', len(rooms_relaxed)
    
    # 3. Relaxed Province
    province_relaxed = df[
        (df["purpose"] == purpose) &
        (df["province_name"] == province) &
        (df["property_type"] == property_type) &
        (df["city"] != city)
    ].copy()
    
    if len(province_relaxed) >= min_results:
        province_relaxed['match_type'] = 'Same Province & Type'
        province_relaxed['price_diff'] = abs(province_relaxed['price'] - predicted_price)
        return province_relaxed.sort_values('price_diff'), 'Same Province', len(province_relaxed)
    
    # 4. Relaxed Type
    type_relaxed = df[
        (df["purpose"] == purpose) &
        (df["city"] == city) &
        (df["property_type"] != property_type)
    ].copy()
    
    if len(type_relaxed) >= min_results:
        type_relaxed['match_type'] = 'Same City, Different Type'
        type_relaxed['price_diff'] = abs(type_relaxed['price'] - predicted_price)
        return type_relaxed.sort_values('price_diff'), 'Same City', len(type_relaxed)
    
    # 5. Last Resort, same purpose only.
    last_resort = df[df["purpose"] == purpose].copy()
    
    if len(last_resort) > 0:
        last_resort['match_type'] = 'General Search'
        last_resort['price_diff'] = abs(last_resort['price'] - predicted_price)
        return last_resort.sort_values('price_diff'), 'Similar Properties', len(last_resort)
    
    return pd.DataFrame(), 'No Match', 0

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
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 Pakistan Property Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get instant price estimates for properties across major Pakistani cities</div>', unsafe_allow_html=True)

if not models_loaded or not data_loaded:
    st.error("⚠️ Cannot start app: Missing required files (models or data).")
    st.stop()

area_min = int( df['area_sqft'].min() )
area_max = int( df['area_sqft'].max() )
bed_max = int( df['bedrooms'].max() )
bath_max = int( df['baths'].max() )
year_min = int( df['year_added'].min() )
year_max = int( df['year_added'].max() )

with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app predicts property prices in Pakistan using ML-models trained on real estate data from major cities using priority-based property real-time matching.
    
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
        help="City will update based on province selection"
    )

with col2:
    st.subheader("Property Type")
    property_type = st.selectbox(
        "Type",
        prop_types,
        help="Type of property"
    )
    year_added = st.slider(
        "Year Listed",
        min_value=year_min,
        max_value=year_max,
        value=year_max,
        step=1,
        help=f"Properties listed between {year_min}-{year_max}"
    )

with col3:
    st.subheader("Specifications")
    # Changed to sliders for checks.
    area_sqft = st.slider(
        "Area (sqft)",
        min_value=area_min,
        max_value=area_max,
        value=1100,
        step=50,
        help=f"Valid range: {area_min:,} - {area_max:,} sqft"
    )
    
    bedrooms = st.slider(
        "Bedrooms",
        min_value=0,
        max_value=min( 15 , bed_max ),
        value=1,
        step=1,
        help=f"Number of bedrooms (0-{bed_max})"
    )
    
    baths = st.slider(
        "Bathrooms",
        min_value=0,
        max_value=min( 15 , bath_max ),
        value=1,
        step=1,
        help=f"Number of bathrooms (0-{bath_max})"
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
                    f"{area_marla:.2f}",
                    help="Property area in marlas (1 Marla = 272.25 sqft)"
                )
            
            with col3:
                room_density = (bedrooms + baths) / area_sqft
                density_label = "Spacious 🏡" if room_density < 0.002 else "Compact 🏢" if room_density > 0.004 else "Standard 🏠"
                st.metric(
                    "Space Rating",
                    density_label,
                    help="Based on room-to-area ratio"
                )
            
            st.divider()
            st.header("🔍 Similar Properties in Market")
            
            with st.spinner("Finding similar properties..."):
                similar_properties, match_label, total_matches = find_similar_properties_by_priority(
                    df, purpose, city, province, property_type, 
                    bedrooms, baths, predicted_price, min_results=5
                )
                
                if not similar_properties.empty:
                    similar_top = similar_properties.head(5)                    
                    similar_top['price_diff_pct'] = (similar_top['price_diff'] / predicted_price) * 100
                    
                    st.success(f"✅ Found {total_matches:,} properties using **{match_label}** criteria (showing top 5)")            

                    display_df = similar_top[[
                        'match_type', 'price', 'city', 'property_type', 
                        'area_sqft', 'bedrooms', 'baths', 'price_diff_pct'
                    ]].copy()
                    display_df['price'] = display_df['price'].apply(lambda x: f"PKR {x:,.0f}")
                    display_df['area_sqft'] = display_df['area_sqft'].apply(lambda x: f"{x:,.0f} sqft")
                    display_df['price_diff_pct'] = display_df['price_diff_pct'].apply(lambda x: f"{x:.1f}%")
                    display_df = display_df.rename(columns={
                        'match_type': 'Match Quality',
                        'price': 'Listed Price',
                        'city': 'City',
                        'property_type': 'Type',
                        'area_sqft': 'Area',
                        'bedrooms': 'Beds',
                        'baths': 'Baths',
                        'price_diff_pct': 'Price Diff'
                    })

                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.subheader("📊 Price Distribution Comparison")                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    prices_millions = similar_top['price'] / 1e6
                    ax.hist(prices_millions, bins=min(20, len(similar_top)), 
                            edgecolor='black', alpha=0.7, color='#1E88E5')                    
                    ax.axvline(predicted_price/1e6, color='red', linestyle='--', 
                                linewidth=2, label='Your Prediction')                    
                    ax.set_xlabel('Price (Millions PKR)', fontsize=12)
                    ax.set_ylabel('Number of Properties', fontsize=12)
                    ax.set_title(f'Price Distribution - {match_label}', fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(alpha=0.3, linestyle='--')                    
                    median_price = similar_top['price'].median()
                    mean_price = similar_top['price'].mean()
                    stats_text = f"Median: PKR {median_price/1e6:.2f}M | Mean: PKR {mean_price/1e6:.2f}M"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ No similar properties found in the dataset. Try adjusting your search criteria.")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏠 <b>Pakistan House Price Predictor</b> | Built with Streamlit & Machine Learning</p>
    <p style='font-size: 0.9rem;'>Data sourced from zameen.com | Models trained on 100K+ properties</p>
</div>
""", unsafe_allow_html=True)