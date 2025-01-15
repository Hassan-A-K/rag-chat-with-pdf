import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

llm_local = ChatOllama(model="yinr/Qwen2.5-agi:latest")

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more PDF files to begin!",
            accept=["application/pdf"],
            max_size_mb=250,
            max_files=100000,
            timeout=3000,
        ).send()

    file_names = ", ".join([file.name for file in files])
    msg = cl.Message(content=f"Processing `{file_names}`...")
    await msg.send()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    texts = []
    metadatas = []

    for file in files:
        pdf = PyPDF2.PdfReader(file.path)
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()

            # Split page text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(page_text)

            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "book": file.name,
                    "page": page_num,  # Add page number metadata
                    "chunk_id": i
                })

    # Create a Chroma vector store with detailed metadata
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Initialize the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing `{file_names}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            # Extract metadata
            book_name = source_doc.metadata.get("book", "Unknown Book")
            page_num = source_doc.metadata.get("page", "Unknown Page")
            chunk_id = source_doc.metadata.get("chunk_id", "Unknown Chunk")
            source_name = f"{book_name} (Page {page_num}"

            # Append text element with the source name
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )

        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()