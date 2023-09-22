import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
MAX_INPUT_SIZE = 200

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    print(llm.model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    
    if len(user_question) > MAX_INPUT_SIZE:
        st.warning(f"Please limit your question to {MAX_INPUT_SIZE} characters.")
    #elif st.session_state.conversation == None:  
    #    st.warning(f"Please upload documents first.")
    else:
        response = st.session_state.conversation({'question':user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)


def main(): 
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs", 
        page_icon=":books:",
    )
    
    st. write(css, unsafe_allow_html=True)

    # App Title and Overview
    st.title("Concussion Recovery AI Chatbot")

    # Add a placeholder image (make sure you have an image named 'concussion_placeholder.jpg' in the working directory or provide a path to it)
    st.image("milad-fakurian-58Z17lnVS4U-unsplash.jpg", caption="Concussion Recovery", width=300)

    



    with st.sidebar:
        st.subheader("This is the sidebar")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
    # Sidebar Navigation
    page = st.sidebar.selectbox("Choose a Page", ["Overview", "Chat with AI", "Sources"])

    if page == "Overview":
        st.header("Overview")
        st.write("""
        Welcome to our specialized AI Chatbot focused on concussion recovery information.
        This chatbot provides insights, recommendations, and answers based on trusted medical sources.
        Navigate to the chat interface to begin your inquiry.
        """)

    elif page == "Chat with AI":
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        
        st.header("Chat Interface")

        user_question = st.text_input("Ask the chatbot a question:")
        if user_question:
            handle_userinput(user_question)
            #st.slider("Rate the response", 0, 5, 3)

    else:  # Sources Page
        st.header("Sources")
        st.write("""
        Our chatbot bases its knowledge on the following trusted sources:
        - Source 1: [Concussion Recovery Guidelines](#)
        - Source 2: [Brain Injury Association Recommendations](#)
        - Source 3: [Neurological Studies on Concussions](#)
        """)

        new_source = st.text_input("Suggest an additional source:")
        if st.button("Submit Source"):
            st.write(f"Thank you for suggesting: {new_source}")

if __name__ == '__main__':
    main()