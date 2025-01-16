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
    files = None  # Initialize variable to store uploaded files

    # Wait for the user to upload files
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more PDF files to begin!",
            accept=["application/pdf"],
            max_size_mb=250,
            max_files=500,
            timeout=300,
        ).send()

    # Inform the user that processing has started
    file_names = ", ".join([file.name for file in files])
    msg = cl.Message(content=f"Processing `{file_names}`...")
    await msg.send()

    # Combine text from all PDF files
    texts = []
    metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in files:
        pdf = PyPDF2.PdfReader(file.path)
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            chunks = text_splitter.split_text(text)
            texts.extend(chunks)
            metadatas.extend([{
                "source": f"{file.name}, Page {page_number}"
            } for _ in chunks])

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file_names}` done. You can now ask questions!"
    await msg.update()

    # Store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_doc in source_documents:
            source_metadata = source_doc.metadata.get("source", "Unknown Source")
            # Create the text element with the source name
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_metadata)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # Return results
    await cl.Message(content=answer, elements=text_elements).send()