import openai, time
import streamlit as st
import utilities as util
import streamlit_ext as ste

# Configure Streamlit page and state
st.set_page_config(page_title="Video Summarizer")
tooltip_style = """
<style>
div[data-baseweb="tooltip"] {
  width: 200px;
}
</style>
"""

# Define functions
def get_df(user_inp, get_transcript, start_time, end_time):
    transcript = get_transcript(user_inp)
    df = util.transcript_to_df(transcript)
    df = util.extract_df(df, start_time, end_time)
    df['summary'] = df['text']
    return df

def add_summary(df, chunk_len, gpt3_setting):
    tmp = df.copy()
    tmp['num_words'] = tmp['summary'].apply(lambda x: len(x.split()))
    tmp['chunk_id'] = tmp['num_words'].cumsum() // int(chunk_len)
    text_chunks, text_summary = [], []
    unique_ids = tmp['chunk_id'].unique()
    my_bar = st.progress(0)
    for idx, cid in enumerate(unique_ids):
        if gpt3_setting['model'] != "text-davinci-003":
            time.sleep(0.5)  # to avoid hitting data transmit cap
        text = tmp[tmp['chunk_id'] == cid]['text']
        text = ' '.join(text.tolist())
        text_chunks.append(text)
        text = tmp[tmp['chunk_id'] == cid]['summary']
        text = ' '.join(text.tolist())
        text = ' '.join(text.split())
        text_summary.append(util.summarize_text(text, gpt3_setting))
        my_bar.progress((idx + 1) / len(unique_ids))
    my_bar.empty()
    tmp = util.get_summary_df(tmp, text_chunks, text_summary)
    return tmp

def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]

def condense_click():
    st.session_state.condense_click = ''

# Render Sidebar
with st.sidebar:
    input_method = st.radio(
        'Select an input method',
        ('Enter a YouTube video ID', 'Upload a Otter transcript file')
    )

    if input_method == 'Enter a YouTube video ID':
        user_inp = st.text_input('YouTube Video ID', placeholder='O1ARBJKspmA')
        get_transcript = util.get_youtube_transcript
    else:
        user_inp = st.file_uploader("Choose a Otter transcript file")
        get_transcript = util.get_otter_transcript

    start_time = st.text_input('Start Timestamp', placeholder='01:07:20')
    end_time = st.text_input('End Timestamp', placeholder='03:02:00')

    key = st.text_input(
        label='OpenAI API Key',
        placeholder='*****************',
        type='password',
        help='''
            I do not store your key. However, if you   
            have security concern, delete the key in   
            your OpenAI account after each use.
        ''',
        key='key'
    )
    openai.api_key = st.session_state.key

    submit_click = st.button('Submit', type="primary")

    st.write('---')
    st.write('Advance Setting')
    model = st.selectbox(
        'Model',
        ('text-davinci-003', 'text-curie-001', 'text-babbage-001', 'text-ada-001'),
    )
    chunk_len = st.slider('Chunk Length', 100, 1000, 500, 100)
    temperature = st.slider('Temperature', 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider('Max Token', 256, 1024, 512, 128)
    top_p = st.slider('Top P', 0.0, 1.0, 1.0, 0.1)
    frequency_penalty = st.slider('Frequency Penalty', 0.0, 1.0, 0.0, 0.1)
    presence_penalty = st.slider('Presence Penalty', 0.0, 1.0, 0.0, 0.1)

    gpt3_setting = dict(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

# Render Output
if 'submit_click' not in st.session_state:
    st.session_state.submit_click = ''
if 'condense_click' not in st.session_state:
    st.session_state.condense_click = ''
if submit_click:
    st.session_state.submit_click = True

if not st.session_state.submit_click:
    st.title('YouTube Video Summarizer')
    st.markdown(
        """
        Hello. I am an AI agent trained to summarize YouTube transcripts. I can help you generate 
        summaries for a YouTube video quickly. I hope that I can save you time, but I must admit 
        that I'm not perfect and I make mistakes. So please do check the summary I wrote before 
        you use it for any purpose. 
          
        Before you begin, please check if the YouTube video contains a transcript. YouTube usually 
        creates a transcript for the video automatically, but it may take some time before it is 
        available after the video is uploaded.         
        
        Use the side bar to enter the video ID, or alternatively, upload a Otter transcript file. 
        Enter the start timestamp and the end timestamp to narrow down the portion of the video 
        where you want the summary.  
        
        You must have a OpenAI API key to use this app. Go to [OpenAI](https://openai.com/api/) to 
        register an account. OpenAI gives you free $18.00 credit for 3 months when you first sign up.
        
        If you run into an error, double check the video ID, and the start/end timestamp. Make sure 
        the API key is entered correctly. Refresh the page to try again.
        """
    )
else:

    if not st.session_state.condense_click:
        if 'summary_df' not in st.session_state:
            df = get_df(user_inp, get_transcript, start_time, end_time)
        else:
            df = st.session_state.summary_df
        with st.spinner(text="Compiling summary. Please be patient..."):
            st.session_state.summary_df = add_summary(df, chunk_len, gpt3_setting)
            summary_csv = st.session_state.summary_df.to_csv(index=False).encode('utf-8')
            output_text_full = util.write_output(
                st.session_state.summary_df,
                summary_only=False,
                file_name=None
            )
            output_text_short = util.write_output(
                st.session_state.summary_df,
                summary_only=True,
                file_name=None
            )
        st.session_state.output_text_full = output_text_full
        st.session_state.output_text_short = output_text_short
        st.session_state.summary_csv = summary_csv
        st.session_state.condense_click = True

    col1, col2, col3 = st.columns(3)
    with col1:
        ste.download_button(
            label="Download data as CSV",
            data=st.session_state.summary_csv,
            file_name='summary.csv',
        )
    with col2:
        ste.download_button(
            label='Download summary full',
            data=st.session_state.output_text_full,
            file_name='summary_full.txt',
        )
    with col3:
        ste.download_button(
            label='Download summary short',
            data=st.session_state.output_text_short,
            file_name='summary_only.txt',
        )
    _, col4, col5 = st.columns(3)
    with col4:
        st.button('Condense More', type="primary", on_click=condense_click)
    with col5:
        st.button('Reset Session', type="primary", on_click=reset_session)

    st.write(st.session_state.output_text_full)
