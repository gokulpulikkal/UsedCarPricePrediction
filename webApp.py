# Import required libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from dateutil.relativedelta import relativedelta
import datetime
import options
import nltk
import re
import time
import base64
nltk.download('opinion_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize


# Saving the default model file path
default_model_path = "randomForest.pkl"

# Constants for the app
loading_gif_path = "l2.gif"
selectbox_color = '#000000'
title_color = '#134F5C'

# Tab title
st.set_page_config(page_title="Let's predict price for vehicle", layout="wide")

# Method for loading the model
def loadModel(modelPath):
    with open(modelPath, "rb") as f:
        model = pickle.load(f)
        return model

# Method to calculate difference in months from the string date
def difference_in_months(strdate):
  strdate = strdate.split('t')[0]
  current_date = datetime.now()
  current_date = current_date.strftime("%Y-%m-%d")
  current_date = datetime.strptime(current_date, "%Y-%m-%d")
  posting_date = datetime.strptime(strdate, "%Y-%m-%d")
  difference_in_months = relativedelta(current_date, posting_date).years * 12 + relativedelta(current_date, posting_date).months
  return difference_in_months

# Function to add custom CSS using a base64 encoded image
def add_custom_css(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# remove the URLs from the text
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# remove the numbers from the text
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# remove the special characters from the text
def remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

# Taking the positive words
positive_words = set(opinion_lexicon.positive())
def extract_positive_words(text):
    words = word_tokenize(text)  # Tokenize the text into words
    positive_words_found = [word for word in words if word in positive_words]
    if len(positive_words_found) == float('nan'):
        return 0
    else:
        return len(positive_words_found)


# Extract negative words
negative_words = set(opinion_lexicon.negative())

# Method to extract negative words from a text
def extract_negative_words(text):
    words = word_tokenize(text)
    negative_words_found = [word for word in words if word in negative_words]
    if len(negative_words_found) == float('nan'):
        return 0
    else:
        return len(negative_words_found)

# Method to get the sentiment of the sentence
def classify_sentiment(p, n):
    if p > n:
        return 1
    else:
        return 0

# Create a function to predict the price
def predict_price(region, year, manuf, car_model, condition, cylinders, fuel, odometer, title_status, trans, drive, size, type, paint, desc, state, posting_date):
    #preprocessing given input by user
    # Columns expected by the model
    df = pd.DataFrame(columns=['year',
 'cylinders',
 'odometer',
 'age',
 'posted_ago_in_months',
 'positive_words',
 'negative_words',
 'region_albuquerque',
 'region_austin',
 'region_baltimore',
 'region_bend',
 'region_boise',
 'region_boston',
 'region_chicago',
 'region_cleveland',
 'region_colorado springs',
 'region_columbus',
 'region_denver',
 'region_des moines',
 'region_detroit metro',
 'region_eugene',
 'region_fresno / madera',
 'region_grand rapids',
 'region_houston',
 'region_jacksonville',
 'region_kansas city, mo',
 'region_las vegas',
 'region_long island',
 'region_los angeles',
 'region_maine',
 'region_milwaukee',
 'region_minneapolis / st paul',
 'region_new hampshire',
 'region_new york city',
 'region_norfolk / hampton roads',
 'region_oklahoma city',
 'region_orange county',
 'region_orlando',
 'region_other',
 'region_philadelphia',
 'region_phoenix',
 'region_pittsburgh',
 'region_portland',
 'region_reno / tahoe',
 'region_rochester',
 'region_sacramento',
 'region_salem',
 'region_sarasota-bradenton',
 'region_seattle-tacoma',
 'region_sf bay area',
 'region_south jersey',
 "region_spokane / coeur d'alene",
 'region_st louis, mo',
 'region_stockton',
 'region_tampa bay area',
 'region_tucson',
 'region_tulsa',
 'region_washington, dc',
 'manufacturer_acura',
 'manufacturer_audi',
 'manufacturer_bmw',
 'manufacturer_cadillac',
 'manufacturer_chevrolet',
 'manufacturer_chrysler',
 'manufacturer_dodge',
 'manufacturer_ford',
 'manufacturer_gmc',
 'manufacturer_honda',
 'manufacturer_hyundai',
 'manufacturer_jeep',
 'manufacturer_kia',
 'manufacturer_lexus',
 'manufacturer_mercedes-benz',
 'manufacturer_nissan',
 'manufacturer_other',
 'manufacturer_ram',
 'manufacturer_subaru',
 'manufacturer_toyota',
 'manufacturer_volkswagen',
 'model_1500',
 'model_2500',
 'model_3500',
 'model_accord',
 'model_altima',
 'model_camaro',
 'model_camry',
 'model_charger',
 'model_civic',
 'model_corolla',
 'model_corvette',
 'model_cr-v',
 'model_cruze',
 'model_edge',
 'model_elantra',
 'model_equinox',
 'model_escape',
 'model_explorer',
 'model_f-150',
 'model_f-250',
 'model_f-350',
 'model_focus',
 'model_forester',
 'model_fusion',
 'model_grand caravan',
 'model_grand cherokee',
 'model_impala',
 'model_jetta',
 'model_malibu',
 'model_mustang',
 'model_odyssey',
 'model_other',
 'model_outback',
 'model_pilot',
 'model_prius',
 'model_rav4',
 'model_rogue',
 'model_sentra',
 'model_sienna',
 'model_sierra',
 'model_sierra 1500',
 'model_silverado',
 'model_silverado 1500',
 'model_silverado 2500hd',
 'model_sonata',
 'model_soul',
 'model_tacoma',
 'model_tahoe',
 'model_tundra',
 'model_wrangler',
 'model_wrangler unlimited',
 'condition_excellent',
 'condition_fair',
 'condition_good',
 'condition_like new',
 'condition_new',
 'condition_salvage',
 'condition_uncharted',
 'fuel_diesel',
 'fuel_electric',
 'fuel_gas',
 'fuel_hybrid',
 'fuel_other',
 'title_status_clean',
 'title_status_lien',
 'title_status_missing',
 'title_status_parts only',
 'title_status_rebuilt',
 'title_status_salvage',
 'transmission_automatic',
 'transmission_manual',
 'transmission_other',
 'drive_4wd',
 'drive_fwd',
 'drive_rwd',
 'drive_uncharted',
 'size_compact',
 'size_full-size',
 'size_mid-size',
 'size_sub-compact',
 'size_uncharted',
 'type_bus',
 'type_convertible',
 'type_coupe',
 'type_hatchback',
 'type_mini-van',
 'type_offroad',
 'type_other',
 'type_pickup',
 'type_sedan',
 'type_suv',
 'type_truck',
 'type_uncharted',
 'type_van',
 'type_wagon',
 'paint_color_black',
 'paint_color_blue',
 'paint_color_brown',
 'paint_color_custom',
 'paint_color_green',
 'paint_color_grey',
 'paint_color_orange',
 'paint_color_purple',
 'paint_color_red',
 'paint_color_silver',
 'paint_color_uncharted',
 'paint_color_white',
 'paint_color_yellow',
 'state_ak',
 'state_al',
 'state_ar',
 'state_az',
 'state_ca',
 'state_co',
 'state_ct',
 'state_dc',
 'state_de',
 'state_fl',
 'state_ga',
 'state_hi',
 'state_ia',
 'state_id',
 'state_il',
 'state_in',
 'state_ks',
 'state_ky',
 'state_la',
 'state_ma',
 'state_md',
 'state_me',
 'state_mi',
 'state_mn',
 'state_mo',
 'state_ms',
 'state_mt',
 'state_nc',
 'state_nd',
 'state_ne',
 'state_nh',
 'state_nj',
 'state_nm',
 'state_nv',
 'state_ny',
 'state_oh',
 'state_ok',
 'state_or',
 'state_pa',
 'state_ri',
 'state_sc',
 'state_sd',
 'state_tn',
 'state_tx',
 'state_ut',
 'state_va',
 'state_vt',
 'state_wa',
 'state_wi',
 'state_wv',
 'state_wy',
 'sentiment_Neutral',
 'sentiment_Positive'])
    current_year = datetime.datetime.now().year
    car_age = current_year - year
    posted_ago = posting_date
    # Processing the description sentiment
    description = remove_urls(desc)
    description = remove_numbers(description)
    description = remove_special_characters(description)
    positive_words = extract_positive_words(description)
    negative_words = extract_negative_words(description)
    sentiment = classify_sentiment(positive_words, negative_words)

    # Creating the data point as expected by the model
    region_col = 'region_'+region
    manuf_col = 'manufacturer_'+manuf
    model_col = 'model_'+car_model
    condition_col = 'condition_'+condition
    fuel_col = 'fuel_'+fuel
    title_status_col = 'title_status_'+title_status
    trans_col = 'transmission_'+trans
    drive_col = 'drive_'+drive
    size_col = 'size_'+size
    type_col = 'type_'+type
    paint_col = 'paint_color_'+paint
    state_col = 'state_'+state

    df.loc[0,region_col] = 1
    df.loc[0,manuf_col] = 1
    df.loc[0,model_col] = 1
    df.loc[0,condition_col] = 1
    df.loc[0,fuel_col] = 1
    df.loc[0,title_status_col] = 1
    df.loc[0,trans_col] = 1
    df.loc[0,drive_col] = 1
    df.loc[0,size_col] = 1
    df.loc[0,type_col] = 1
    df.loc[0,paint_col] = 1
    df.loc[0,state_col] = 1
    df.loc[0,'cylinders'] = int(cylinders)
    df.loc[0,'year'] = int(year)
    df.loc[0,'age'] = int(car_age)
    df.loc[0,'odometer'] = int(odometer)
    df.loc[0,'sentiment_Positive'] = int(sentiment)
    df.loc[0,'posted_ago_in_months'] = int(posted_ago)
    df.loc[0,'positive_words'] = int(positive_words)
    df.loc[0,'negative_words'] = int(negative_words)
    column_names = [region_col, manuf_col, model_col, condition_col, fuel_col, title_status_col, trans_col, drive_col, size_col, type_col, paint_col, state_col, 'year', 'age', 'odometer','sentiment_value','posted_ago_in_months','positive_words','negative_words', 'cylinders']

    for col in df.columns:
        if col not in column_names:
            df.loc[0,col] = 0
    
    dummy_sample = df

    # Make a prediction using the model
    prediction = model.predict(dummy_sample)
    return prediction

# Aligning input fields
col1, col2, col3 = st.columns([0.1, 1, 0.1])
# Banner image
col2.image("bgt1.jpeg")
st.title("") #Don't remove. This is added for extra space. Empty is not seems to be working

# Alignment of input fields
col1, col2, col3 = st.columns([3, 3, 3], gap="large")

# Read user input from the input fields
with col1:
    year = st.number_input("Manufacturing Year", min_value=1980, max_value=2023)
with col1:
    cylinder = st.number_input("Number of cylinders", min_value=3, max_value=8)
with col1:
    odometer = st.number_input("Odometer reading (miles)", min_value=0)
with col1:
    manufacturer = st.selectbox("Select Manufacturer *", index=None,options=options.manufacturer_options, placeholder="Select Manufacturer")
with col1:
    region = st.selectbox("select Region *",options=options.regions_options, index=None, placeholder="Select Region")
with col1:
    selected_state = st.selectbox("Choose state *", options.state, index=None, placeholder="Select State")
with col2:
    condition = st.selectbox("Please select Condition of your vehicle",options=options.vehicle_condition_options, index=None, placeholder="Vehicle condition")
with col2:
    fuel = st.selectbox("Select fuel type of your vehicle *",options=options.fuel_options, index=None, placeholder="Fuel type")
with col2:
    title_status = st.selectbox("Select title status of your vehicle *",options=options.title_status_options, index=None, placeholder="Title Status")
with col2:
    transmission = st.selectbox("Select transmission type of your vehicle *",options=options.transmission_options, index=None, placeholder="Transmission")
with col2:
    vehicle_size = st.selectbox("Please select size type of your vehicle",options=options.vehicle_sizes, index=None, placeholder="Vehicle size")
with col3:
    vehicle_color = st.selectbox("Please select color of your vehicle",options=options.colors, index=None, placeholder="Vehicle color")
with col3:
    months_ago_number = st.number_input("Posted months ago", min_value=0)
with col3:
    selected_model = st.selectbox("Choose car model *", options.models, index=None, placeholder="Vehicle Model")
with col3:
    selected_drive_type = st.selectbox("Please Choose car drive type", options.drive_options, index=None, placeholder="Vehicle Drive type")
with col3:
    selected_type = st.selectbox("Please Choose car type", options.type, index=None, placeholder="Vehicle type")


description = st.text_area("Describe your vehicle")

# if user does not give any input for non-required fields
# Making the default value saved
if not selected_drive_type:
    selected_drive_type = "uncharted"
if not vehicle_size:
    vehicle_size = "uncharted"
if not selected_type:
    selected_type = "uncharted"
if not vehicle_color:
    vehicle_color = "uncharted"
if not condition:
    condition = "uncharted"
    
# Make the prediction
col1, col2 = st.columns(2)

# Displaying results
if st.button("Predict Price"):
    if not (fuel and title_status and transmission and selected_model and selected_state and manufacturer and region):
        st.warning("Please fill all the required fields *")
    else: 
        # Reading the saved variables
        manufacturer = str(manufacturer).lower()
        region = str(region).lower()
        condition = str(condition).lower()
        fuel = str(fuel).lower()
        title_status = str(title_status).lower()
        transmission = str(transmission).lower()
        vehicle_size = str(vehicle_size).lower()
        vehicle_color = str(vehicle_color).lower()
        selected_model = str(selected_model).lower()
        selected_drive_type = str(selected_drive_type).lower()
        selected_type = str(selected_type).lower()
        selected_state = str(selected_state).lower()

        # Getting the model object
        model = loadModel(default_model_path)
        image_container = st.empty()
        resultContainer = st.empty()
        # Loading indicator
        image_container.image(loading_gif_path, width=480)
        # Putting fake loading 
        time.sleep(4)
        predicted_price = predict_price(region, year, manufacturer, selected_model, condition, cylinder, fuel, odometer, title_status, transmission, selected_drive_type, vehicle_size, selected_type, vehicle_color, description, selected_state, months_ago_number)
        if predicted_price < 0:
            predicted_price = 0
        image_container.empty()
        resultContainer.write(f"Price predicted for used car is: {predicted_price[0]} $")
        icol1, icol2 = st.columns([1.99,2.11])
        icol2.image("feature_importances_rf.png")
        icol1.image("pricevsage.png")
