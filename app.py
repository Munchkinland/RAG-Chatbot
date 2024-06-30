import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader, UnstructuredPDFLoader, UnstructuredWordLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set your API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the following context from the document: {context}\n\nAnswer the question: {question}"
)

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.html *.pdf *.docx *.txt")])
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".html":
            loader = UnstructuredHTMLLoader(file_path=file_path)
        elif file_extension == ".pdf":
            loader = UnstructuredPDFLoader(file_path=file_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordLoader(file_path=file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path=file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format")
            return

        global car_docs
        car_docs = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        car_docs_split = text_splitter.split_documents(car_docs)

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create FAISS vector store from document chunks
        global vectorstore
        vectorstore = FAISS.from_documents(car_docs_split, embeddings)

        # Create retriever
        global retriever
        retriever = vectorstore.as_retriever()

        messagebox.showinfo("Success", "File loaded and processed successfully")

def ask_question():
    query = question_entry.get()
    if not query:
        messagebox.showerror("Error", "Please enter a question")
        return

    if 'retriever' not in globals():
        messagebox.showerror("Error", "Please load a document first")
        return

    # Define RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt_template": prompt_template,
        },
    )

    # Invoke RAG chain with the query
    result = rag_chain({"question": query})
    answer = result["result"]

    # Display the answer
    answer_display.delete(1.0, tk.END)
    answer_display.insert(tk.END, answer)

# Create GUI
root = tk.Tk()
root.title("Document Q&A")

# Load file button
load_button = tk.Button(root, text="Load Document", command=load_file)
load_button.pack(pady=10)

# Question entry
question_label = tk.Label(root, text="Enter your question:")
question_label.pack()
question_entry = tk.Entry(root, width=50)
question_entry.pack(pady=5)

# Ask question button
ask_button = tk.Button(root, text="Ask Question", command=ask_question)
ask_button.pack(pady=10)

# Answer display
answer_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
answer_display.pack(pady=10)

# Start the GUI event loop
root.mainloop()
