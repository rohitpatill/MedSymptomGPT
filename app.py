import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set the browser tab title and layout
st.set_page_config(page_title="MedSymptomGPT", page_icon="🩺", layout="wide")

# Load the model and tokenizer
model_path = 'MedSymptomGPT.pt'
tokenizer = GPT2Tokenizer.from_pretrained('distilbert/distilgpt2')
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Function to generate symptoms based on disease name
def generate_symptoms(disease_name):
    inputs = tokenizer.encode(disease_name, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    symptoms = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return symptoms

# Streamlit UI
# Add a logo
st.image("ai.png", width=100)

st.title("Medical Symptom Predictor 🩺")
st.write("**Enter a disease name to get its symptoms:**")

# Add some spacing and a divider
st.markdown("---")

# Text input for disease name
disease_name = st.text_input("Disease Name", placeholder="e.g., Diabetes, Hypertension")

# Add a button to submit
if st.button("Submit"):
    if disease_name:
        with st.spinner("Generating symptoms..."):
            symptoms = generate_symptoms(disease_name)
            st.markdown("### Predicted Symptoms")
            st.success(symptoms)
    else:
        st.error("Please enter a disease name.")

# Add a sidebar for additional information or links
st.sidebar.title("About MedSymptomGPT")
st.sidebar.info(
    """
    MedSymptomGPT is a machine learning application using DistilGPT-2, fine-tuned on the Diseases_Symptoms dataset.
    This tool predicts medical symptoms based on disease names using advanced natural language processing techniques.
    """
)
st.sidebar.markdown("### Useful Links")
st.sidebar.markdown("[Streamlit Documentation](https://docs.streamlit.io/)")
st.sidebar.markdown("[GitHub Repository](https://github.com/rohitpatill/MedSymptomGPT)")
st.sidebar.markdown("### Contact")
st.sidebar.markdown("For issues or feedback, please contact [rohitp2001k@gmail.com](mailto:rohitp2001k@gmail.com)")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: left;">
        <p><strong>MedSymptomGPT</strong></p>
        <p>Developed by <a href="https://www.linkedin.com/in/rohitpatill" target="_blank">Rohit Patil</a></p>
      
       
    </div>
    """, unsafe_allow_html=True)
























# import streamlit as st
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # Set the browser tab title
# st.set_page_config(page_title="MedSymptomGPT")

# # Load the model and tokenizer
# model_path = 'MedSymptomGPT.pt'
# tokenizer = GPT2Tokenizer.from_pretrained('distilbert/distilgpt2')
# model = torch.load(model_path, map_location=torch.device('cpu'))
# model.eval()

# # Function to generate symptoms based on disease name
# def generate_symptoms(disease_name):
#     inputs = tokenizer.encode(disease_name, return_tensors='pt')
#     outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
#     symptoms = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return symptoms

# # Streamlit UI
# st.title("Medical Symptom Predictor")
# st.write("Enter a disease name to get its symptoms:")

# # Text input for disease name
# disease_name = st.text_input("Disease Name")

# if st.button("Submit"):
#     if disease_name:
#         with st.spinner("Generating symptoms..."):
#             symptoms = generate_symptoms(disease_name)
#             st.write("### Predicted Symptoms")
#             st.write(symptoms)
#     else:
#         st.error("Please enter a disease name.")


#     #Run the app with following command in terminal
#     #streamlit run app.py