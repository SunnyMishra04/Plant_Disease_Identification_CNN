import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌱")

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
import numpy as np
import google.generativeai as genai

# Configure Gemini API using environment variable
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize app mode in session state
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Function to query Gemini AI for crop-related questions
def ask_ai_crop_question(question):
    try:
        # Use Gemini Pro model for text generation
        model = genai.GenerativeModel('gemini-pro')
        
        # Create a prompt with context for crop-related questions
        full_prompt = f"""You are an expert agricultural AI assistant. 
        Provide a helpful and informative answer to the following crop management or plant disease question:
        {question}"""
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Sorry, I couldn't get a response from the AI."

# Home Page
if st.session_state.app_mode == "Home":
    st.title("🌱 Smart Crop Management🌱")
    st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 15px 32px;
            margin: 10px;
            border: none;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

    # Check if home page image exists
    if os.path.exists("home_page.jpeg"):
        st.image("home_page.jpeg", use_container_width=True)
    else:
        st.warning("Home page image not found. Please ensure 'home_page.jpeg' is in the correct directory.")

    st.markdown("""
    ### Welcome to the Smart Crop Management! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    """)

    # Add navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Disease Detection", key="disease_detection_btn", use_container_width=True):
            st.session_state.app_mode = "Disease Recognition"
    
    with col2:
        if st.button("🤖 AI Crop Guide", key="ai_crop_guide_btn", use_container_width=True):
            st.session_state.app_mode = "AI Crop Assistant"

    st.markdown("""
    ### How It Works:
    1. **Disease Detection:** Upload an image of a plant with suspected diseases, and our system will analyze it.
    2. **AI Crop Guide:** Ask questions about crop management and get expert AI-powered advice.

    ### Get Started
    Click on the buttons above to explore our Plant Disease Recognition System!
    
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif st.session_state.app_mode == "About":
    st.title("About This Project")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. 
    This dataset consists of about 2000+ RGB images of healthy and diseased crop leaves, 
    categorized into 38 different classes. The dataset is split into an 80/20 ratio 
    for training and validation sets, preserving the directory structure.

    #### Content:
    1. **train** (2197 images)
    2. **test** (33 images)
    3. **validation** (550 images)
    """)

# Disease Recognition Page
elif st.session_state.app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    st.markdown("""
    Upload an image of a plant leaf to check if it has any diseases.
    """, unsafe_allow_html=True)

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, use_container_width=True)

    # Predict button
    if st.button("Predict"):
        
        st.write("Our Prediction")

        # Ensure that the image is saved temporarily for prediction
        if test_image is not None:
            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(test_image.getbuffer())
            
            # Perform prediction on the temporary saved image
            result_index = model_prediction("temp_image.jpg")
            
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            # Display result
            st.success(f"Model Prediction: **{class_name[result_index]}**")
        else:
            st.error("Please upload an image to predict the disease.")

# Crop Management Blog Page
elif st.session_state.app_mode == "Crop Management Blog":
    st.title("🌾 Crop Management Blog 🌾")

    # Introduction to the Blog Section
    st.markdown("""
    Welcome to our blog section! Here, you can find useful tips and articles on crop management, 
    pest control, irrigation, and more. Stay tuned for the latest articles on improving your 
    farming practices.
    """)

    # Blog Titles (can be fetched from a file or database later)
    blog_titles = [
        "Crop Rotation Tips for Better Yield", 
        "How to Identify Pests Early", 
        "Watering Techniques for Healthy Plants", 
        "Best Organic Fertilizers for Your Crops"
    ]

    # Allow users to select a blog post
    selected_blog = st.selectbox("Choose a Blog Post:", blog_titles)

    # Display selected blog content
    if selected_blog == blog_titles[0]:
        st.markdown("""
        ### Crop Rotation Tips for Better Yield
        Crop rotation is one of the most effective practices to improve soil fertility and control pests. 
        By changing the crops planted in a specific field each year, you reduce the buildup of pests 
        and diseases, and promote healthier soil.
        Crop rotation is one of the most effective practices to improve soil fertility and control pests. By changing the crops planted in a specific field each year, you reduce the buildup of pests and diseases, and promote healthier soil. This practice allows the soil to rest and replenish nutrients, reducing the need for chemical fertilizers. Crop rotation also helps to maintain biodiversity and balance in the ecosystem.

For example, alternating between legumes, such as peas or beans, and other crops like corn or wheat, allows nitrogen-fixing plants to naturally replenish soil nitrogen levels. This minimizes the need for artificial fertilizers. In addition, crops like tomatoes and cabbage, which are susceptible to similar pests, should not be grown in the same area consecutively.

Key Benefits of Crop Rotation:

    Improved soil structure and fertility
    Reduced soil erosion
    Control of soil-borne diseases and pests
    Better management of weeds
    Increased biodiversity in the farm ecosystem
        """)
    elif selected_blog == blog_titles[1]:
        st.markdown("""
        ### How to Identify Pests Early
        Early pest detection is crucial for minimizing damage to your crops. Learn how to identify 
        the common pests affecting your crops and take proactive measures.Early pest detection is crucial for minimizing damage to your crops. Learn how to identify the common pests affecting your crops and take proactive measures. Pests can damage plants by feeding on leaves, stems, or roots, leading to poor growth and reduced yields. They can also spread diseases that can devastate crops.

Start by regularly inspecting your plants for signs of damage or unusual symptoms. Some common pests to watch out for include aphids, caterpillars, whiteflies, and beetles. Keep an eye out for discolored or deformed leaves, holes in leaves, and presence of sticky residue (often a sign of aphids).

Pest Control Tips:

    Introduce natural predators, such as ladybugs, to your garden
    Use insecticidal soaps or neem oil for a natural pesticide
    Remove affected plant parts immediately to prevent the pests from spreading
    Ensure your crops are properly spaced to allow air circulation, which can deter pests

By monitoring your plants regularly and acting early, you can prevent pest problems from getting out of hand and protect your crops from significant damage.
        """)
    elif selected_blog == blog_titles[2]:
        st.markdown("""
        ### Watering Techniques for Healthy Plants
        Efficient watering is key to growing healthy crops. Discover the best irrigation methods 
        for different types of plants, including drip irrigation and soaker hoses.Efficient watering is key to growing healthy crops. Discover the best irrigation methods for different types of plants, including drip irrigation and soaker hoses. Over-watering or under-watering can lead to poor growth, root rot, or dehydration, making it essential to water your plants in a way that meets their specific needs.

Drip irrigation is one of the most effective methods for water-efficient farming. It delivers water directly to the base of each plant, reducing water wastage and preventing fungal diseases caused by wet foliage. Another great option is soaker hoses, which allow water to seep gently into the soil, ensuring deep and even moisture distribution.

Watering Tips:

    Water in the early morning or late evening to reduce evaporation
    Ensure your soil is well-drained to prevent waterlogging
    Use mulch around your plants to retain moisture and keep the soil cool
    Adjust watering schedules based on the weather conditions, particularly during hot or dry spells

By using the right irrigation techniques and paying attention to the water needs of your crops, you can ensure they grow strong and healthy while conserving water resources.
        """)
    else:
        st.markdown("""
        ### Best Organic Fertilizers for Your Crops
        Organic fertilizers are a sustainable way to improve soil quality and enhance plant growth. 
        This article covers the best organic fertilizers for various crops.Organic fertilizers are a sustainable way to improve soil quality and enhance plant growth. They are derived from natural sources like animal manure, compost, and plant-based materials, making them a safer alternative to chemical fertilizers. Organic fertilizers help to improve soil structure, retain moisture, and promote the healthy development of beneficial microorganisms.

Some of the best organic fertilizers include compost, worm castings, and well-rotted manure. Compost is rich in nutrients and provides a slow-release source of nourishment for plants. Worm castings, often referred to as "black gold," contain beneficial microbes that improve soil health. Manure from cows, chickens, or horses is also an excellent source of nutrients, but it must be well-aged to avoid burning plants.

Organic Fertilizer Application Tips:

    Apply fertilizers in the spring and fall for best results
    Spread fertilizers evenly around the base of the plant, being careful not to touch the roots directly
    Mix organic fertilizers into the soil to ensure even distribution
    Use mulch on top of the soil after fertilizing to keep the nutrients in place

Using organic fertilizers not only improves the health of your plants but also helps sustain the long-term health of your soil. By choosing natural methods, you contribute to a more sustainable and environmentally friendly farming practice.
        """)

# AI Crop Assistant
elif st.session_state.app_mode == "AI Crop Assistant":
    st.title("Ask AI Your Crop Questions 🌾🤖")
    question = st.text_input("Ask a question about crop management or plant diseases:(e.g., 'How do I treat tomato blight?')")
    
    if question:
        answer = ask_ai_crop_question(question)
        st.write("AI Answer:")
        st.write(answer)

# Sidebar Navigation
if st.sidebar.button("🏠 Home"):
    st.session_state.app_mode = "Home"
if st.sidebar.button("ℹ️ About"):
    st.session_state.app_mode = "About"
if st.sidebar.button("🔍 Disease Recognition"):
    st.session_state.app_mode = "Disease Recognition"
if st.sidebar.button("📝 Crop Management Blog"):
    st.session_state.app_mode = "Crop Management Blog"
if st.sidebar.button("🤖 AI Crop Assistant"):
    st.session_state.app_mode = "AI Crop Assistant"