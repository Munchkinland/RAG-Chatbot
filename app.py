import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for, flash
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, StuffDocumentsChain

# Set your API key
openai_api_key = os.environ.get("OPENAI_API_KEY", "tu_openai_api_key_aqui")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the following context from the document: {context}\n\nAnswer the question: {question}"
)

app = Flask(__name__)
app.secret_key = "supersecretkey"
retriever = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            try:
                # Save file to a temporary location
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, file.filename)
                file.save(temp_path)

                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension == ".html":
                    loader = UnstructuredHTMLLoader(file_path=temp_path)
                elif file_extension == ".pdf":
                    loader = UnstructuredPDFLoader(file_path=temp_path)
                elif file_extension == ".docx":
                    loader = UnstructuredWordDocumentLoader(file_path=temp_path)
                elif file_extension == ".txt":
                    loader = TextLoader(file_path=temp_path)
                else:
                    flash("Unsupported file format", "error")
                    return redirect(url_for("index"))

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

                flash("File loaded and processed successfully", "success")
                return redirect(url_for("ask"))
            except Exception as e:
                flash(str(e), "error")
                return redirect(url_for("index"))

    return render_template("index.html")

@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        question = request.form["question"]
        if retriever is None:
            flash("Please load a document first", "error")
            return redirect(url_for("index"))

        try:
            # Define RAG chain
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={
                    "document_chain": StuffDocumentsChain(prompt_template=prompt_template),
                },
            )

            # Invoke RAG chain with the query
            result = rag_chain({"question": question})
            answer = result["result"]

            return render_template("ask.html", answer=answer)
        except Exception as e:
            flash(str(e), "error")
            return redirect(url_for("ask"))

    return render_template("ask.html", answer="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



