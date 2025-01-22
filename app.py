import streamlit as st
from assistant import MedicalReportAssistant

def main():
    st.title("MedReport.AI")
    
    if 'assistant' not in st.session_state:
        st.session_state.assistant = MedicalReportAssistant()
        st.session_state.document_processed = False
    

    inputFile = st.file_uploader("Upload a PDF of your medical report.", type=['pdf'])
    
    if inputFile and not st.session_state.document_processed:
        with st.spinner("Processing document. Please wait up to 2 minutes:"):
            num_chunks = st.session_state.assistant.processDocument(inputFile)
            st.session_state.document_processed = True


    if st.session_state.document_processed:
        if 'summary' not in st.session_state:
            st.session_state.summary = st.session_state.assistant.retrieveSummary()

        st.write("Summary: \n", st.session_state.summary)
        userQuery = st.text_input("Query the medical report:")
        if userQuery:
            with st.spinner("Looking through your report..."):
                response = st.session_state.assistant.query(userQuery)
                st.write("Answer: ", response)

if __name__ == "__main__":
    main()
