from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from getpass import getpass
import cachetools
import os

# Llave de OpenAI
OPENAI_API_KEY = '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'
os.environ['OPENAI_API_KEY'] = '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'

# Almacenar en caché
cache = cachetools.LRUCache(maxsize=1000)

# Ruta del PDF
current_directory = os.path.dirname(__file__)
pdf_path = os.path.join(current_directory, 'descripcion_proyecto.pdf')

# Cargar datos una vez durante la inicialización de la aplicación
ml_paper = []
loader = PyPDFLoader(pdf_path)
data = loader.load()
ml_paper.extend(data)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
documents = text_splitter.split_documents(ml_paper)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)


def question_process(question):
    if question in cache:
        return cache[question]

    answer = qa_chain.run(question)

    cache[question] = answer

    return answer


class ChatBotView(APIView):
    def post(self, request):
        question = request.data.get('question')

        answer = question_process(question)

        return Response({'Respuesta': answer})

