import streamlit as st
import os
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
from pathlib import Path
from streamlit_chat import message
from llama_index_utils import ReportPulseAssistent
import json
from prompts import REPORT_PROMPT
from PyPDF2 import PdfReader
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qrcode
import webbrowser  # Import the webbrowser module
from PIL import Image
import io



from streamlit import components



st.set_page_config(
    page_title="Report Pulse",
    page_icon="favicon.ico",
)

path = os.path.dirname(__file__)

# Load translations from JSON file
with open(path+"/Assets/translations.json") as f:
    transl = json.load(f)

# Trick to preserve the state of your widgets across pages
for k, v in st.session_state.items():
    st.session_state[k] = v 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create upper navigation buttons like a menu
page = st.selectbox("Menu", ["ReportPulse", "Stats For Nerd"])

# Load the CSV file
@st.cache_data
def load_data():
    data = pd.read_csv("severity_records.csv")
    return data

data = load_data()





def show_visualization():
    # Title and description
    st.title("CSV Dataset Visualizer")
    st.write("This Streamlit app displays data and various visualizations from the CSV file.")
    
    # Load the CSV file
    @st.cache_data  # Caching to improve performance
    def load_data():
        data = pd.read_csv("severity_records.csv")
        return data
    
    data = load_data()
    
    st.header("CSV Data")
    st.write(data)

    # Visualization options
    visualization_type = st.selectbox("Select Visualization Type", ["Pie Chart", "Bar Chart", "Histogram"])

    if visualization_type == "Pie Chart":
        st.subheader("Severity Distribution (Pie Chart)")
        severity_counts = data["Severity"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(severity_counts, labels=severity_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)  # Pass the figure to st.pyplot
    elif visualization_type == "Bar Chart":
        st.subheader("Severity Distribution (Bar Chart)")
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x="Severity", palette="Set1")
        plt.xticks(rotation=45)
        st.pyplot()
    elif visualization_type == "Histogram":
        st.subheader("Histogram of Severity")
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data, x="Severity", bins=len(data["Severity"].unique()), kde=True, color="blue")
        plt.xticks(rotation=45)
        st.pyplot()
    elif visualization_type == "Heatmap":
        st.subheader("Heatmap")
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        st.pyplot()
    





def show_overview():
# Based on the selected menu page, show the corresponding content











    styl = f"""
    <style>
        .stTextInput {{
        position: fixed;
        bottom: 3rem;
        z-index: 2;
        }}
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa {{
            display: flex !important;
            float:left;
            overflow-y: auto;
            flex-direction: column-reverse;
        }}
    </style>
    """
    st.markdown(styl, unsafe_allow_html=True)


    # Add the language selection dropdown    
    if 'lang_tmp' not in st.session_state:
        st.session_state['lang_tmp'] = 'English'

    if 'lang_changed' not in st.session_state:
        st.session_state['lang_changed'] = False

    if 'lang_select' in st.session_state:
        #st.sidebar.markdown("<h3 style='text-align: center; color: black;'>{}</h3>".format(transl[st.session_state['lang_select']]["language_selection"]), unsafe_allow_html=True)
        lang = st.sidebar.selectbox(transl[st.session_state['lang_select']]["language_selection"], options=list(transl.keys()), key='lang_select')
    else:
        #st.sidebar.markdown("<h3 style='text-align: center; color: black;'>{}</h3>".format(transl[st.session_state['lang_tmp']]["language_selection"]), unsafe_allow_html=True)
        lang = st.sidebar.selectbox(transl[st.session_state['lang_tmp']]["language_selection"], options=list(transl.keys()), key='lang_select')

    if lang != st.session_state['lang_tmp']:
        st.session_state['lang_tmp'] = lang
        st.session_state['lang_changed'] = True
        st.experimental_rerun()
    else:
        st.session_state['lang_changed'] = False

    st.title(transl[lang]["title"])

    file_uploaded = st.file_uploader(label=transl[lang]["title"])

    styl = f"""
    <style>
        .stTextInput {{
        position: fixed;
        bottom: 3rem;
        }}
    </style>
    """
    st.markdown(styl, unsafe_allow_html=True)


    if 'csv_file_created' not in st.session_state:
        st.session_state.csv_file_created = False

    csv_file_name = "severity_records.csv"

    # Create a CSV file if it doesn't exist
    if not st.session_state.csv_file_created:
        with open(csv_file_name, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["ID", "Severity"])  # Write the header row
        st.session_state.csv_file_created = True




      

    

    if file_uploaded is not None:
        
        def display_messages():
            
            for i, (msg, is_user) in enumerate(st.session_state["messages"]):
                #message( msg, is_user=is_user, key=str(i), allow_html = True )
                message( msg, is_user=is_user, key=str(i) )
            st.session_state["thinking_spinner"] = st.empty()
        
        def process_input():
            if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
                user_text = st.session_state["user_input"].strip()
                with st.session_state["thinking_spinner"], st.spinner(transl[lang]["thinking"]):
                    agent_text = generate_response(user_text)

                st.session_state["messages"].append((user_text, True))
                st.session_state["messages"].append((agent_text, False))
                st.session_state["user_input"] = ""
                
        # Storing the chat
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        def showmessage(output):
            st.session_state["messages"].append((output, False))

        def validate_value_in_range(record):
            parameter_value = float(record['Result'])
            biological_range = record['Biological Ref Range'].split(' ')[0]
            biological_low_range, biological_high_range  = [float(val) for val in biological_range.split('-')]
            if parameter_value < biological_low_range:
                record['variation'] = str(format(parameter_value - biological_low_range,".2f"))
            elif parameter_value > biological_high_range:
                record['variation'] = str(format(parameter_value - biological_high_range,".2f"))
            return record


        def get_relevant_report(reports):
            # resports data format is expected to be a list of json object
            abnormal_data_numeric = []
            string_data = []
            for record in reports:
                if record.get("Result", "unknown") != "unknown":
                    try:
                        new_record = validate_value_in_range(record)
                        abnormal_data_numeric.append(new_record)
                    except Exception as e:
                        # if here means the value is string type
                        #st.success(e)
                        string_data.append(record)
            new_report = {
                "numeric": abnormal_data_numeric,
                "string": string_data
            }
            return new_report 

        def get_st_col_metric(report):
            
            reports_temp = get_relevant_report(report)['numeric']
            #st.success(json.dumps(reports_temp))
            #st.success(json.dumps(report))
            reports = []
            for rec in reports_temp: 
                if "variation" in rec:
                    reports.append(rec)
                    #reports.remove(rec)

            #st.success(json.dumps(reports))
            # pick the first 5 reports
            reports = reports[:5]
            cols = st.columns(len(reports))
            for col, record in zip(cols, reports):
                col.metric(record["Parameter"], record["Result"], str(record["variation"]))


                
        def upload_file(uploadedFile):
            
            # Save uploaded file to 'content' folder.
            save_folder = '/'
            save_path = Path(save_folder, uploadedFile.name)
            
            with open(save_path, mode='wb') as w:
                w.write(uploadedFile.getvalue())

            with st.spinner(transl[lang]["scan"]):
                return ReportPulseAssistent(save_folder,lang=lang)
            
        reportPulseAgent = upload_file(file_uploaded)
        with st.spinner(transl[lang]["gen_summary"]):    
            r_response = reportPulseAgent.get_next_message(lang=lang,prompt_type='summary')
        
        st.sidebar.markdown(r_response)
        



        
        st.sidebar.markdown(""" <br /><br />
                        :rotating_light: **{}** :rotating_light: <br />
                                {}
                                """.format(transl[lang]['caution'], transl[lang]['caution_message']), 
                                unsafe_allow_html=True
        )
    
        with st.spinner(transl[lang]["gen_report"]): 
            try:
                reports_response = reportPulseAgent.get_next_message(REPORT_PROMPT,lang=lang,prompt_type='report')
                #st.success(json.dumps(reports_response))
                reports = json.loads(reports_response)
                #st.success(json.dumps(reports))
                st.sidebar.markdown(get_st_col_metric(reports))
            except Exception as e:
                pass

        def generate_response(user_query):
            response = reportPulseAgent.get_next_message(user_query, lang=lang,prompt_type='other')
            return response

        # We will get the user's input by calling the get_text function

        st.text_input(transl[lang]['ask_question'],key="user_input", on_change=process_input)
        display_messages()

        # Load the pre-trained BERT model and tokenizer
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        # Sample senior doctor database
        senior_doctors = {
            1: "Dr .Sahil Jadhav",
            2: "Dr .Sanjay Prajapati"
        }

        # Sample junior doctor database
        junior_doctors = {
            101: "Alice",
            102: "Bob",
            103: "Carol"
        }

        treatment_room = {
            "Emergency": {
                "Available": 100,
                "Taken": 3,
            },
            "ICU": {
                "Available": 100,
                "Taken": 4,
            },
            "GeneralWard": {
                "Available": 100,
                "Taken": 6,
            }
        }

        # Function to check the availability of a treatment room
        def check_room_availability(room_type):
            if room_type in treatment_room:
                available_rooms = treatment_room[room_type]["Available"]
                taken_rooms = treatment_room[room_type]["Taken"]
                return available_rooms, taken_rooms
            else:
                return None, None

        # Function to admit a patient to a treatment room
        def admit_patient(room_type):
            if room_type in treatment_room and treatment_room[room_type]["Available"] > 0:
                treatment_room[room_type]["Available"] -= 1
                treatment_room[room_type]["Taken"] += 1
                return True
            else:
                return False


        # Function to allocate a seat based on severity
        def allocate_seat(severity):
            severity = severity.lower()  # Convert severity to lowercase for case-insensitive comparison
            if severity == "high":
                # Allocate a senior doctor
                doctor_id = 1
                doctor_name = senior_doctors.get(doctor_id)
                room_type = "Emergency"
                available_rooms, taken_rooms = check_room_availability(room_type)
                if available_rooms > 0:
                    admit_patient(room_type)
                    return f"{doctor_name} (Doctor-ID: {doctor_id}) will treat you at {room_type} in Bed Number {taken_rooms + 1} "

            elif severity == "medium":
                # Allocate a junior doctor
                doctor_id = 101
                doctor_name = junior_doctors.get(doctor_id)
                room_type = "ICU"
                available_rooms, taken_rooms = check_room_availability(room_type)
                if available_rooms > 0:
                    admit_patient(room_type)
                    return f"{doctor_name} (Doctor-ID: {doctor_id}) will treat you at {room_type} in Bed Number {taken_rooms + 1} "

            elif severity == "low":
                # Allocate a junior doctor
                doctor_id = 101
                doctor_name = junior_doctors.get(doctor_id)
                room_type = "GeneralWard"
                available_rooms, taken_rooms = check_room_availability(room_type)
                if available_rooms > 0:
                    admit_patient(room_type)
                    return f"{doctor_name} (Doctor-ID: {doctor_id}) will treat you at {room_type} in Bed Number {taken_rooms + 1} "

            return "No allocation made."



                    
        def predict_severity(report_text):
            # Tokenize the text
            inputs = tokenizer(report_text, return_tensors='pt', truncation=True, padding=True)

            # Perform inference to classify severity
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            # Map predicted class to severity label
            severity_labels = ['Low', 'Medium', 'High']
            predicted_severity = severity_labels[predicted_class]

            return predicted_severity

        def extract_text_from_pdf(pdf_file_path):
            text = ''
            pdf_reader = PdfReader(pdf_file_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        # Path to the PDF file
        pdf_file_path = file_uploaded
        # Extract text from the PDF file
        pdf_text = extract_text_from_pdf(pdf_file_path)

        # Get the severity prediction for the extracted text
        severity_prediction = predict_severity(pdf_text)

        # Replace the following line in your code
        # print(f"Predicted Severity: {severity_prediction}")

        # Use st.write to display the predicted severity in Streamlit
        st.write(f"Predicted Severity: {severity_prediction}")
        # Check the predicted severity and display the appropriate button in the sidebar

        
       

        # ... (your existing code here)

        # Check the predicted severity and display the appropriate button in the sidebar


        # ... (your existing code here)

        # Check the predicted severity and display the appropriate button in the sidebar


        # ...

        # ... Your previous code ...

        if severity_prediction in ['Low', 'Medium', 'High']:
            if st.button('Book a Doctor'):
                allocation_result = allocate_seat(severity_prediction)
                st.write(allocation_result)

               



                # Code to run when the High Severity button is clicked
        st.markdown("""---""")        
            

        # Write the ID and severity to the CSV file
        with open(csv_file_name, mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                next_id = len(open(csv_file_name).readlines())  # Calculate the next ID
                csv_writer.writerow([next_id, severity_prediction]) 


if page == "ReportPulse":
    show_overview()
elif page == "Stats For Nerd":
    show_visualization()
    
