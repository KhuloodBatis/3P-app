import streamlit as st
import pandas as pd
from io import StringIO

import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import streamlit as st
import cohere
import altair as alt
from PIL import Image

# image = Image.open('assets/logo.png')

path = 'data/ecg.csv'
sample_rate = 500
hr_column_name = 'heartrate'


patient = {
    'age': 51,
    'weight_kg': 90,
    'height_cm': 196,
    'disease': 'congestive heart failure',
    'medication': ['Beta blocker', 'ACE Inhibitors'],
    'gender': 'male',
    'comorbiidities': ['diabetes mellitus', 'obstructive and central apnea syndrom'],
}

def generate_prompt(
        measures,
        # age,
        # gender,
        # disease,
        # weight,
        # height,
        # comorbiidities,
        # medications,
        # bpm,
        # ibi,
        # sdnn,
        # sdsd,
        # rmssd,
        # pnn20,
        # pnn50,
        # hr_mad,
        # sd1,
        # sd2,
        # s,
        # sd1_sd2,
        # breathingrate,
        **kwargs):
    return f'''
Based on the diganosis and measures listed below, create six recommendations for the patient. Each recommendation should be no more than 200 words.

There is a {age} year old {gender} with {disease} disease.

The patient's weight and height are {weight} kg and {height} cm, respectively.

The patient has the following comorbiidities:
{comorbidities}

The patient takes the following medications:
{medications}

The patient has the following heart rate signal measures:
* beats per minute = {measures['bpm']:.2f} BPM
* interbeat interval = {measures['ibi']:.2f} IBI 
* standard deviation if intervals between adjacent beats (SDNN) = {measures['sdnn']:.2f}
* standard deviation of successive differences between adjacent R-R intervals (SDSD) = {measures['sdsd']:.2f}
* root mean square of successive differences between adjacend R-R intervals (RMSSD) = {measures['rmssd']:.2f}
* proportion of differences between R-R intervals greater than 20ms, pNN20 = {measures['pnn20']:.2f}
* proportion of differences between R-R intervals greater than 50ms, pNN50 = {measures['pnn50']:.2f}
* median absolute deviation (MAD) = {measures['hr_mad']:.2f}
* Poincare analysis (SD1) = {measures['sd1']:.2f}
* Poincare analysis (SD2) = {measures['sd2']:.2f}
* Poincare analysis (S) = {measures['s']:.2f}
* Poincare analysis (SD1/SD2) = {measures['sd1/sd2']:.2f}
* breathingrate = {measures['breathingrate']:.2f}
'''.strip()

def generate_analysis_prompt(
        measures,
        **kwargs):
    return f'''
Comment on the severity of every measure provided

The patient has the following heart rate signal measures:
* beats per minute = {measures['bpm']:.2f} BPM
* interbeat interval = {measures['ibi']:.2f} IBI 
* standard deviation if intervals between adjacent beats (SDNN) = {measures['sdnn']:.2f}
* standard deviation of successive differences between adjacent R-R intervals (SDSD) = {measures['sdsd']:.2f}
* root mean square of successive differences between adjacend R-R intervals (RMSSD) = {measures['rmssd']:.2f}
* proportion of differences between R-R intervals greater than 20ms, pNN20 = {measures['pnn20']:.2f}
* proportion of differences between R-R intervals greater than 50ms, pNN50 = {measures['pnn50']:.2f}
* median absolute deviation (MAD) = {measures['hr_mad']:.2f}
* Poincare analysis (SD1) = {measures['sd1']:.2f}
* Poincare analysis (SD2) = {measures['sd2']:.2f}
* Poincare analysis (S) = {measures['s']:.2f}
* Poincare analysis (SD1/SD2) = {measures['sd1/sd2']:.2f}
* breathingrate = {measures['breathingrate']:.2f}
'''.strip()


def generate_report_prompt(
        measures,
        **kwargs):
    return f'''
Explain the medical condition in simple language for this patient. You have 200 words max.

There is a {age} year old {gender} with {disease} disease.

The patient's weight and height are {weight} kg and {height} cm, respectively.

The patient has the following comorbiidities:
{comorbidities}

The patient takes the following medications:
{medications}

The patient has the following heart rate signal measures:
* beats per minute = {measures['bpm']:.2f} BPM
* interbeat interval = {measures['ibi']:.2f} IBI 
* standard deviation if intervals between adjacent beats (SDNN) = {measures['sdnn']:.2f}
* standard deviation of successive differences between adjacent R-R intervals (SDSD) = {measures['sdsd']:.2f}
* root mean square of successive differences between adjacend R-R intervals (RMSSD) = {measures['rmssd']:.2f}
* proportion of differences between R-R intervals greater than 20ms, pNN20 = {measures['pnn20']:.2f}
* proportion of differences between R-R intervals greater than 50ms, pNN50 = {measures['pnn50']:.2f}
* median absolute deviation (MAD) = {measures['hr_mad']:.2f}
* Poincare analysis (SD1) = {measures['sd1']:.2f}
* Poincare analysis (SD2) = {measures['sd2']:.2f}
* Poincare analysis (S) = {measures['s']:.2f}
* Poincare analysis (SD1/SD2) = {measures['sd1/sd2']:.2f}
* breathingrate = {measures['breathingrate']:.2f}
'''.strip()


def generate_severity_prompt(measures, **kwargs):
    return f'''
There is a {age} year old {gender} with {disease} disease.

The patient's weight and height are {weight} kg and {height} cm, respectively.

The patient has the following comorbiidities:
{comorbidities}

The patient takes the following medications:
{medications}

The patient has the following heart rate signal measures:
* beats per minute = {measures['bpm']:.2f} BPM
* interbeat interval = {measures['ibi']:.2f} IBI 
* standard deviation if intervals between adjacent beats (SDNN) = {measures['sdnn']:.2f}
* standard deviation of successive differences between adjacent R-R intervals (SDSD) = {measures['sdsd']:.2f}
* root mean square of successive differences between adjacend R-R intervals (RMSSD) = {measures['rmssd']:.2f}
* proportion of differences between R-R intervals greater than 20ms, pNN20 = {measures['pnn20']:.2f}
* proportion of differences between R-R intervals greater than 50ms, pNN50 = {measures['pnn50']:.2f}
* median absolute deviation (MAD) = {measures['hr_mad']:.2f}
* Poincare analysis (SD1) = {measures['sd1']:.2f}
* Poincare analysis (SD2) = {measures['sd2']:.2f}
* Poincare analysis (S) = {measures['s']:.2f}
* Poincare analysis (SD1/SD2) = {measures['sd1/sd2']:.2f}
* breathingrate = {measures['breathingrate']:.2f}

Describe the severity of the patient's condition using only either mild, moderate, or severe.
'''.strip()

# def read_hr_data(path, hr_column_name, delimiter=','):
#     df = pd.read_csv(path, delimiter=delimiter)
#     data = df[hr_column_name].values
#     return data

def read_hr_data(hr_column_name):
    data = st.session_state['df'][hr_column_name].values
    return data

def submit():    
    co = cohere.Client('2sa2XUEyeYZh5WaLXNIvskXKaOGygnvmzdbnzqzC') # This is your trial API key
    # print(uploaded_file)
    # df = pd.read_csv(uploaded_file, delimiter=';')
    # print(df)
    data = read_hr_data('heartrate')
    _, measures = hp.process(data, sample_rate = sample_rate)
    response = co.generate(
      model='command',
      prompt= generate_prompt(measures, **st.session_state),
      max_tokens=300,
      temperature=0.9,
      k=0,
      stop_sequences=[],
      return_likelihoods='NONE')
    severity = co.generate(
      model='command',
      prompt= generate_severity_prompt(measures, **st.session_state),
      max_tokens=300,
      temperature=0.9,
      k=0,
      stop_sequences=[],
      return_likelihoods='NONE')
    report = co.generate(
      model='command',
      prompt= generate_report_prompt(measures, **st.session_state),
      max_tokens=300,
      temperature=0.9,
      k=0,
      stop_sequences=[],
      return_likelihoods='NONE')
    analysis = co.generate(
      model='command',
      prompt= generate_analysis_prompt(measures, **st.session_state),
      max_tokens=300,
      temperature=0.9,
      k=0,
      stop_sequences=[],
      return_likelihoods='NONE')
    st.session_state['output'] = response.generations[0].text
    st.session_state['severity'] = severity.generations[0].text
    st.session_state['report'] = report.generations[0].text
    st.session_state['analysis'] = analysis.generations[0].text
    st.session_state['df_measures'] = pd.DataFrame.from_dict(measures, orient='index')
    # return response.generations[0].text


# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = 'Not AI-generated yet.'
if 'severity' not in st.session_state:
    st.session_state['severity'] = 'Not AI-generated yet.'
if 'report' not in st.session_state:
    st.session_state['report'] = 'Not AI-generated yet.'
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = 'Not AI-generated yet.'
if 'df_measures' not in st.session_state:
    st.session_state['df_measures'] = ''

st.markdown("<h1 style='text-align: center;'>3P</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>predict, prevent, personalize</h3>", unsafe_allow_html=True)
# st.title('3P')
# st.subheader('predict, prevent, personalize')
# st.image(image, width=200)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.write(' ')

# with col2:
#     st.image(image, width=200)

# with col3:
#     st.write(' ')


# st.subheader('Heart rate signal')
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:

#     # Can be used wherever a "file-like" object is accepted:
#     dataframe = pd.read_csv(uploaded_file, delimiter=';')
#     st.write(dataframe)

# with st.form(key='patient_form'):
# c1, c2 = st.col(2)

# with c2:
st.subheader('🩺 Heart rate signal')
uploaded_file = st.file_uploader("Upload your smartwatch heartrate data.")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    st.session_state['df'] = pd.read_csv(uploaded_file, delimiter=';')
    c = alt.Chart(st.session_state['df'].loc[:4999]).mark_line().encode(
        x='ms',
        y='heartrate'
        )
    st.altair_chart(c, use_container_width=True)
    # st.write(c)
else:
    st.session_state['df'] = pd.read_csv(path, delimiter=';')
    # color='#70DBB3'
    c = alt.Chart(st.session_state['df'].loc[:4999]).mark_line().encode(
        x='ms',
        y='heartrate'
        )
    st.altair_chart(c, use_container_width=True)
#     # st.write(c)

# with c1:
st.subheader('🤒 Patient information')
with st.form(key='patient_form'):
    c1, c2 = st.columns(2)
    with c1:
        age = st.text_input('Age', value=patient['age'], key='age')
        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            key="gender"
            )
        weight = st.text_input('Weight', value=patient['weight_kg'], key='weight')
        height = st.text_input('Height', value=patient['height_cm'], key='height')
    with c2:
        disease = st.selectbox(
            "Disease",
            ("Congestive heart failure", "Coronary heart disease", "Right heart failure", "Pericarditis"),
            )
        comorbidities = st.multiselect(
            'Comorbidities',
            ['Diabetes', 'Hypertension', 'COPD'],
            ['Diabetes', 'Hypertension'],)
        medications = st.multiselect(
            'Medications',
            ['Benazepril', 'Captopril', 'Enalapril', 'Fosinopril'],
            ['Benazepril', 'Captopril'],)
    st.form_submit_button('Submit', on_click=submit)
    



st.subheader('🤯 Severity')
# progress_text = "Severity"
# st.markdown("""
# <style>
# .stProgress .st-bo {
#     background-color: red;
# }
# </style>
# """, unsafe_allow_html=True)
# st.progress(75)
color_code = {
    'severe' : '🟥',
    'moderate': '🟨',
    'mild' : '🟩',
    'Not AI-generated yet.':'⬜',
}
st.info(st.session_state['severity'].strip(), icon=color_code[st.session_state['severity'].strip()])

st.subheader('❤️‍🩹 Heartrate analysis')
st.write(st.session_state['df_measures'])
st.write(st.session_state.analysis)

st.subheader('👨‍⚕️ Diagnosis')
st.write(st.session_state.report)

st.subheader('💊 Recommendations')
st.write(st.session_state.output)
