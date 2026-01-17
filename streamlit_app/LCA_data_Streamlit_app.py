import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from natsort import natsorted
from joblib import load
import PyPDF2
import pymupdf4llm
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import Chroma  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.output_parsers import CommaSeparatedListOutputParser


def load_pdf_paths(pdf_directory, base_persist_directory):
    """Load PDF file paths in natural order and assign persist directories in groups of three with local IDs (1,2,3) per batch."""
    
    files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
    sorted_files = natsorted(files)  
    
    all_files = [str(Path(pdf_directory, file).as_posix()) for file in sorted_files]
    
    num_of_papers = len(all_files)  

    document_ids = list(range(1, num_of_papers + 1))

    persist_directories = [
        str(Path(base_persist_directory, f"db_{doc_id}").as_posix())
        for doc_id in document_ids
    ]
    
    df_pdf = pd.DataFrame({
        'Document_ID': document_ids, 
        'PDF_path': all_files, 
        'Vectordb_path': persist_directories
    })
    
    return df_pdf
    
def load_the_document(pdf_path):
    """Loads the document and extracts text content"""

    # Load document
    docs = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    # Extract metadata
    paper_title = docs[0]['metadata'].get('title', 'Unknown Title')
    num_pages = docs[0]['metadata'].get('page_count', 0)
    author = docs[0]['metadata'].get('author', 'Unknown Author')

    pages_content = []

    for i, page in enumerate(docs, start=1):
        page_text = remove_reference(page['text']).strip()
        pages_content.append({
            "page_number": i,
            "text": page_text
        })

    return pages_content, paper_title, author, num_pages


def remove_reference(pdf_text):
    """Removes references and acknowledgment sections, along with inline citations, from a given PDF text."""

    # Remove references and acknowledgment sections entirely
    pdf_text = re.split(r'(?i)\bReferences\b|\bAcknowledgment[s]?\b', pdf_text)[0].strip()

    # Remove inline citations like [1], [12-15], (Smith et al., 2021), etc.
    citation_patterns = [
        r'\[[^\]]*\d{1,4}\]',          # numeric citations like [1], [12], [ABC12]
        r'\([^\)]*et al\.,?\s*\d{4}[^\)]*\)',  # (Author, 2020)
        r'\(\s*\d{4}[a-z]?\s*\)',  # (2021a)
    ]

    for pattern in citation_patterns:
        pdf_text = re.sub(pattern, '', pdf_text)

    # Clean up extra spaces and blank lines
    pdf_text = '\n'.join(line.strip() for line in pdf_text.splitlines() if line.strip())

    return pdf_text


def embedding_document(pages_content, ID, persist_directory, openai_api_key):
    """Embeds documents and stores them persistently in a vector database (Chroma), recording page number."""

    documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,   # small for better retrieval precision
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", "."]
    )

    document_id = ID

    for page in pages_content:
        page_number = page["page_number"]
        page_text = page["text"]

        # Split this page into smaller chunks
        page_chunks = text_splitter.create_documents([page_text])

        for chunk in page_chunks:
            documents.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "page_number": page_number,
                        "document_id": document_id 
                    }
                )
            )

    num_chunks = len(documents)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings, 
        persist_directory=persist_directory
    )

    vectordb.persist()

    return num_chunks

def build_vectorDB(df_path, openai_api_key):
    """Processes each PDF, extracts metadata, builds vector DB, and updates DataFrame."""
   
    success_status = []  # Track success for each PDF
    paper_titles = []  # Store paper titles
    author_list = []
    num_pages_list = []  # Store number of pages
    chunk_nums = []  # Store chunk numbers separately

    for _, row in df_path.iterrows():
        pdf_path = row["PDF_path"]
        ID = row["Document_ID"]
        persist_directory = row["Vectordb_path"]
        
        try:
            # Extract content, title, and page count
            docs, paper_title, author, num_pages = load_the_document(pdf_path)

            # Build the vector database
            num_chunks = embedding_document(docs, ID, persist_directory, openai_api_key)  # Returns an integer

            success_status.append(True)

            # Store extracted metadata
            paper_titles.append(paper_title)
            author_list.append(author)
            num_pages_list.append(num_pages)
            chunk_nums.append(num_chunks) 

            print(f"Successfully processed {pdf_path}")
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            success_status.append(False)  # Mark as failed

            # Store placeholders for failed cases
            paper_titles.append(None)
            author_list.append(None)
            num_pages_list.append(None)
            chunk_nums.append(None)  
            vectordb = None  # Ensure vectordb is always defined

    # Add extracted metadata to the DataFrame
    df_path["Paper_title"] = paper_titles
    df_path["Author"] = author_list
    df_path["Num_pages"] = num_pages_list
    df_path["Num_chunks"] = chunk_nums  
    df_path["Is_added_to_vectorDB"] = success_status

    return df_path

def df_to_csv(df, file_name):
    """Write a DataFrame to a CSV file"""
    df.to_csv(file_name, index=False, escapechar='\\')

def LCA_information_extraction(document_id, persist_directory, openai_api_key):
    """Extract the system boundary, functional unit, target product, LCIA methods, impact category, LCIA results, and geography from the given paper"""
    
    #create a Chroma vector store, specifying the persistence directory
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docstorage = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    k = docstorage._collection.count()
    retriever = docstorage.as_retriever(search_kwargs={"filter": {"document_id": document_id}, "k": k})
    
    #initial the LLM model
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=4000,
        openai_api_key=openai_api_key
    )
    
    system_prompt = (
        """
        You are a helpful assistant specializing in extracting life cycle assessment (LCA) data. 
        Based on the provided context, use the definitions below to answer the question at the end. 
        If you are unsure of the answer, state "I don't know" rather than conjecture.
        """
    )
    
    context = """
        **System boundary**: is defined as the scope of the LCA, which specifies the life cycle stages and processes that are considered in LCA studies. Common system boundaries are cradle-to-gate, gate-to-gate, and cradle-to-grave:
            - Cradle-to-gate: from the raw material extraction (cradle) up to the factory gate, includes raw material extraction, transport, and manufacturing.
            - Cradle-to-grave: covers the entire product lifecycle from raw material extraction (cradle) to the final disposal (grave). 
            - Gate-to-gate: only includes the production stage.
        **Functional unit (FU)**: is a quantified description of the performance of a product. The purpose of the functional unit is to provide a reference to which the inputs and outputs can be related.
        **Reference product**: is the main output of a product system that delivers the function described by the functional unit. 
        **Life Cycle Impact Assessment (LCIA) method**: is defined as the method to quantify the environmental impacts of a product system. It analyzes the data from the inventory phase and translates it into meaningful impact categories. The common LCIA method includes IPCC, ReCiPe, TRACI, CML, EF and Impact 2002+.
        **Impact category**: is a set of categories that represent a different type of potential environmental impact of a product system. These impact categories are selected based on the chosen impact assessment method. The common impact categories include global warming potentials, ozone depletion, eutrophication, acidification, human toxicity, and photochemical ozone formation. 
        **Life cycle assessment (LCIA) results**: a quantitative environment impacts of a product. They can be presented as a specific value or a range, and are typically with units according to selected impact categories, such as global warming potential (kg CO₂-eq), acidification potential (kg SO₂-eq), and eutrophication potential (kg PO₄³⁻-eq).
        **Geography**: refers to the specific physical or regional context (e.g., country, continent, or specific site) in which the LCA study is conducted or modeled. 
        
        ### Instructions for Data Extraction: 
        - Identify and list all information with minimum detail to understand their relationships. 
        - Is really IMPORTANT to extract all data from the given article and ensure completeness and precision in your extraction. 
        - Extract all impact categories from the article, such as global warming potentials, ozone depletion, eutrophication, acidification, human toxicity, and photochemical ozone formation.
        - Extract only LCIA results related to the impact category of global warming potential (GWP), climate change (CC), or GHG emissions, which use CO₂-eq as units. Represent LCIA results in the format: ["LCIA results": "Reference producs/Scenarios/Cases: :Impact values + Units"].
        - Avoid extra explanations and format your response as a structured JSON format as follows:
          [{{"System boundary": "", "Functional unit": "", "Impact assessment method": "", "Impact category": [], "LCIA results": [], "Geography": ""}}]
        - If you cannot find information about the system boundary, functional unit, reference product, impact assessment method, LCIA results, or geography, label them as "Not mentioned."
    
        ###Examples:
        1. Context: <The "cradle to grave" is defined as the system boundary of all the six comparison scenarios, which covers the material and energy production chain and all processes from the raw material extraction through the production, transportation, and use phase up to the product's end of life treatment. In this study, the functional unit is defined as 1 kg and 1 MJ of H2 carrier produced from coal, natural gas, and renewables. The socalled CML 2001 method is applied to LCIA calculation. 
        LCA results and impact analysis\n\n\nGHG emissions\nProduction phase (This study) Reference\n\nCoal-CH 3 OH (kg 3.09 2.6 to 3.8 [39]\nCO 2 -eq/kg)\n\nNG-CH 3 OH 0.84 0.873 to 0.881 [40]\n(kg CO 2 -eq/kg)\n\n\nPV/CCU-CH 3 OH 1.04 0.99 [41]\n(kg CO 2 -eq/kg)\n\nCoal-NH 3 3.93 3.85 [22]\n(kg CO 2 -eq/kg)\n\nNG-NH 3 2.70 2.74 [13]\n(kg CO 2 -eq/kg)\n\nPV-NH 3 0.78 0.93 [42]\n(kg CO 2 -eq/kg)
        This work is based on the nation conditions of China.>
        Answer: 
            [{{"System boundary": "Cradle-to-grave",
              "Functional unit": "1 kg and 1 MJ of H2 carrier produced from coal, natural gas, and renewables",
              "Reference product": "Methanol and Ammonia",
              "Impact assessment method": "CML 2001",
              "Impact category": [
                "Global warming potential (GWP)",
                "Acidification potential",
                "Ozone depletion potential",
                "Photochemical oxidant creation potential",
                "Eutrophication potential",
                "Abiotic depletion potential"],
              "LCIA results": [
                "Coal-CH3OH: 3.09 kg CO2-eq/kg",
                "NG-CH3OH: 0.84 kg CO2-eq/kg",
                "PV/CCU-CH3OH: 1.04 kg CO2-eq/kg",
                "Coal-NH3: 3.93 kg CO2-eq/kg",
                "NG-NH3: 2.70 kg CO2-eq/kg",
                "PV/CCU-NH3: 0.78 kg CO2-eq/kg"],
                "Geography": "China"}}]
            
        2. Context: <As indicated in Fig. 1, the scope of the study is from ‘cradle to gate’, with two main stages considered: biomass supply (cultivation, collection and transportation to the processing plant); and production of bio-ethylene and its co-products. The functional unit is defined as the production of 1 tonne of ethylene. The SimaPro v.8.3. software (Pré Consultants B.V., 2017) has been used for the life cycle modelling and the impacts have been calculated following the CML 2 method (Guinée et al., 2001), using the April 2016 update.
        Case 3 is the best option\nwith the negative net values for these three categories: -62.4 GJ/t\n(ADP fossil ), -0.07 t CO 2 eq./kg (GWP), and -59 mg CFC-11 eq./t\n(ODP).
        The production plant is assumed to be based in the Duero Valley (Castilla y Leon, Spain) as there is extensive cultivation of poplar there due to favourable climatic conditions.>
        Answer:
        [{{"System boundary": "Cradle-to-gate",
            "Functional unit": "1 tonne of ethylene",
            "Reference product": "Ethylene",
            "Impact assessment method": "CML 2",
            "Impact category": [
              "Global warming potential (GWP)"],
            "LCIA results": [
              "Case 3: -0.07 t CO2-eq./kg (GWP)"],
            "Geography": "Duero Valley (Castilla y Leon, Spain)"}}]
        3. Context: <The LCA model has a cradle-to-gate scope, and the system boundary includes sugarcane farming, bagasse transportation, size reduction, pretreatment, enzymatic hydrolysis, fermentation, and downstream separation, as shown in Fig. 2. 1 kg of lactic acid is the functional unit. OpenLCA 1.9 is used to create a product system. ReCiPe 2016 methodology with the hierarchist perspective, commonly used in LCA literature (Hiloidhari et al., 2020), is implemented.
        The total life cycle climate change impact for production of 1 kg of lactic acid was 4.62 kg CO 2 eq.
        The goal of the LCA is to quantify the environmental impacts of bagasse based LA production facility annexed with an Indian sugar mill.>
        Answer:
        [{{"System boundary": "Cradle-to-gate",
            "Functional unit": "1 kg of lactic acid",
            "Reference product": "Lactic acid",
            "Impact assessment method": "ReCiPe 2016",
            "Impact category": [
              "Global warming potential (GWP)"],
            "LCIA results": [
              "Lactic acid: 4.62 kg CO2-eq./kg"],
            "Geography": "India"}}]
    """
    
    # Create a chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", context),
            ("user", "Response to the question: {question}")
        ]
    )
    
    chain_type_kwargs = {
        "prompt": prompt_template,
        "document_variable_name": "question"
    }
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs
    )
    
    question = "Extract the system boundary, functional unit, reference product, impact assessment method, impact category, LCIA results, or geography for me."
    result = qa_chain({"query": question})

    content = result.get("result", "").strip()

    # Remove triple backticks and optional 'json' specifier
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    # Parse to Python object
    try:
        data_list = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw content: {content}")

    return data_list

def process_document(df_vectordb, openai_api_key):
    """Processing each paper for data extraction"""

    results = []

    for _, row in df_vectordb.iterrows():  # Iterate over all_files
        document_id = row["Document_ID"]
        persist_directory = row["Vectordb_path"]
        PDF_path = row["PDF_path"]
        
        try:

            # Extract information
            LCA_data = LCA_information_extraction(document_id, persist_directory, openai_api_key)

            # Add metadata columns
            for entry in LCA_data:
                entry["Author"] = row["Author"]  # Add Author from df_vectordb
                entry["Paper_title"] = row["Paper_title"]  # Add Paper title
                entry["PDF_path"] = row["PDF_path"]  # Add PDF path
            
            results.append(LCA_data)
            print(f"Successfully processed document: {PDF_path}")

        except Exception as e:
            print(f"Error processing processed document: {PDF_path}: {e}")
            continue  # Skip to the next document in case of an error

    # Flatten nested lists
    all_result = [item for sublist in results for item in sublist]

    return all_result

def csv_to_df(file_name):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_name)

def df_to_csv(df, file_name):
    """Write a DataFrame to a CSV file"""
    df.to_csv(file_name, index=False, escapechar='\\')

def extract_tables_and_pages(pdf_path):
    """Extract table titles from the paper"""
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        data = []  # This will store tuples of (pdf_path, title, page)

        # Regular expression to find table titles
        table_title_regex = re.compile(r'[Tt][Aa][Bb][Ll][Ee]+\.?+\s*[A-Z]?\d+\.?\s+[A-Z]+.*')

        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                found_titles = table_title_regex.findall(text)
                for title in found_titles:
                    data.append((pdf_path, title, i + 1))  # Append the path, title, and page number

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['PDF_path', 'Table_title', 'Page'])
    df['Table_title'] = df['Table_title'].str.replace(r'\n', ' ', regex=True)
    return df

def extract_file_name_from_path(file_path):
    """Extract the file name"""
    
    file_name_with_extension = file_path.split('/')[-1]
    # Remove the file extension
    title = file_name_with_extension.replace('.pdf', '')
    return title

def generate_embeddings(text, openai_api_key):
    """Initial the embedding model"""
    
    
    client = OpenAI(api_key=openai_api_key)
    
    # make api call
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    # return text embedding
    return response.data

def load_pdf_paths_table(pdf_directory):
    """Load PDF file paths in natural order and assign persist directories in groups of three with local IDs (1,2,3) per batch."""
    
    files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
    sorted_files = natsorted(files)  
    
    all_files = [str(Path(pdf_directory, file).as_posix()) for file in sorted_files]
    
    return all_files

def LCI_pathway_extraction(table_title, page_number, persist_directory, openai_api_key):
    """Extract the pathway and activity from the inventory table"""

    #create a Chroma vector store, specifying the persistence directory
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"filter": {"page_number": page_number}})

    #initial the llm
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=4000,
        openai_api_key=openai_api_key
    )

    #initial the memory
    msgs = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    #initial the prompt
    template1 = """
        You are a helpful assistant specializing in extracting life cycle assessment (LCA) data. Based on the provided context, use the definitions below to answer the question at the end. If you are unsure of the answer, state "I don't know" rather than conjecture.

        ### Extract the information based on the definitions below: 
        - **Pathway:** The sequence of processes and transformations that raw materials undergo to become a final product. This includes all intermediate steps, inputs, and outputs. Synonyms include "Processes," "Technical route," "Scenario," "Conversion." 
        - **Activity:** A unit process of converting raw materials into intermediates or products. It represents the smallest element in the product system for which input and output data are quantified. Examples include raw material production, transport, assembly, synthesis, production, and end-of-life treatment. 

        ### Instructions for Data Extraction: 
        - Identify and list the pathways and activities with minimum detail to understand their relationships. 
        - Only extract the pathway and activity from the given table's meta. Ensure completeness and precision in your extraction. 
        - Avoid extra explanations and format your response as JSON format follows: {{"Pathway": ["Activity"], "Pathway": ["Activity"]}}
            - If you do not find any information about the pathway or activity, label them as "Not mentioned."

        ###Examples:
        1. Table: <The inventory data for PV/CCU-CH 3 OH technical route\n\nInputs Values Units Outputs Values Units\n\n**Hydrogen generation**\nElectricity 10070 kWh Hydrogen 190 kg\n\n
        Water 1710 MJ Waste water 87.4 kg\n\nPotassium hydroxide 0.57 kg Oxygen 1501 kg\n\nHydrogen plant 1.02E-6 Unit\n\n**Methanol synthesis and purification**\nElectricity 42.75 kWh Methanol 1000 kg\n\nHydrogen 190 MJ Waste water 630 kg\n\n
        Carbon dioxide (HP) 1455 kg\n\nAluminium oxide 0.012 kg\n\nCopper oxide 0.062 kg\n\nZinc oxide 0.029 kg>
        Answer: 
        - “Pathway”: [“PV/CCU-CH 3 OH technical route”]
        - “Activity”: [“Hydrogen generation”, “Methanol synthesis and purification”]
        2. Table: <Table. Life Cycle Energy Consumption Inventory Data of the CtEG Route for Producing 1 ton EG\n\nparameter unit coal mining and processing coal transportation coal to EG total life cycle\n\nMaterials Consumption [8]\n\ncoal t 3.17 × 10 [0] 3.17 × 10 [0]\n\n
        Energy Consumption [8] [,] [22] [,] [31]\n\nfuel coal GJ 2.89 × 10 [0] 4.73 × 10 [1] 5.02 × 10 [1]\n\ndiesel GJ 5.79 × 10 [−] [2] 2.47 × 10 [−] [1] 3.05 × 10 [−] [1]\n\ngasoline GJ 5.79 × 10 [−] [2] 1.73 × 10 [−] [2] 7.52 × 10 [−] [2]\n\nelectricity GJ 5.79 × 10 [−] [1] 1.47 × 10 [−] [1] 3.16 × 10 [0] 3.89 × 10 [0]\n\n
        Pollutant Emission [8] [,] [22] [,] [32]\n\nCO 2 kg 1.26 × 10 [2] 6.30 × 10 [1] 6.72 × 10 [3] 6.91 × 10 [3]\n\nCH 4 kg 2.37 × 10 [0] 3.42 × 10 [−] [1] 2.44 × 10 [1] 2.71 × 10 [1]\n\nN 2 O kg 2.13 × 10 [−] [3] 1.47 × 10 [−] [4] 7.69 × 10 [−] [3] 9.97 × 10 [−] [3]\n\nCO kg 1.51 × 10 [−] [2] 5.41 × 10 [−] [1] 1.06 × 10 [1] 1.12 × 10 [1]\n\n
        NO x kg 4.67 × 10 [−] [1] 1.17 × 10 [−] [1] 3.53 × 10 [0] 4.11 × 10 [0]\n\nSO 2 kg 1.69 × 10 [−] [1] 2.69 × 10 [−] [2] 1.44 × 10 [0] 1.64 × 10 [0]\n\nPM 10 kg 5.23 × 10 [−] [1] 1.08 × 10 [−] [1] 2.92 × 10 [0] 3.55 × 10 [0]\n\nVOC kg 8.24 × 10 [−] [3] 8.00 × 10 [−] [3] 8.58 × 10 [−] [2] 1.02 × 10 [−] [1]\n\n>
        Answer: 
        - “Pathway”: [“CtEG Route”]
        - “Activity”: [“Coal mining and processing”, “coal mining and processing”, “ Coal to EG”]
        3. Tanle: <Table.** Inventory to produce 1 metric ton of EG via different pathways[14, 18]\n\nUnit Petro-EG Coal-EG\n\nFeedstock\n\nCoal t 3.17\n\nMethanol t 0.60\n\nNH 3 t 0.30\n\nEthylene t 0.75\n\nWater t 3\n\nUtilites\n\nElectricity kWh 300 1200\n\n4.2 MPa steam t 0.10 0.42\n\n
        1.5 MPa steam t 0.40 4.50\n\n0.5 MPa steam t 0.10 3.90\n\nDirect CO 2 emissions t 0.94 5.44\n\n17\n>
        Answer:
        - "Pathway": ["Petro-EG", "Coal-EG"]
        - "Activity": [ ]

        {chat_history}
        {context}
        Question: {question}
        """

    prompt1 = PromptTemplate(
        template=template1, input_variables=["context", "chat_history", "question"], output_parser=CommaSeparatedListOutputParser()
    )

    #set up chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt1}
    )

    #get the result
    result1 = []

    query1 = f"extract the Pathway and Activity from the [{table_title}]."
    result = chain({"question": query1})
    # Parse the model's string output into a Python dictionary
    extracted = result["answer"].strip('```json').strip('```').strip()
    try:
        result_dict = json.loads(extracted)
    except json.JSONDecodeError:
        result_dict = {"Pathway": ["Not mentioned"], "Activity": ["Not mentioned"]}

    return result_dict, retriever

def LCI_data_extraction(table_title, page_number, pathway, activity, persist_directory, openai_api_key):
    """Extract LCI data from the inventory table"""

    #create a Chroma vector store, specifying the persistence directory
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"filter": {"page_number": page_number}})
    
    #initial the llm
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=4000,
        openai_api_key=openai_api_key
    )

    #initial the memory
    msgs = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    #initial the prompt
    template2 = """
        You are a helpful assistant specializing in extracting life cycle assessment (LCA) data. Based on the provided context, use the definitions below to answer the question at the end. If you are unsure of the answer, state "I don't know" rather than conjecture.

        ### Extract the information based on the definitions below: 
        - **Inputs**: is defined as input substances used to manufacture intermediates or products. Input materials generally include human-made systems and resources from the environment. The examples of inputs are feedstocks, energy, natural resources, intermediates, devices, transports, solvents, and chemicals. These input materials are quantified by a numeric value with a unit.
        - **Outputs**: is defined as output substances from the production activity. Output materials typically include emissions, wastes, and output products. These outputs are also quantified using specific numeric values with units.

        ### Instructions for data Extraction: 
        - Identify and list the inputs and outputs with minimum detail to understand their relationships. 
        - Inputs include both human-made products like raw materials, chemicals, intermediates, products, components, devices, solvents, electricity,  fuel and resources from the environment like water.
        - Outputs mainly include emissions to air, emissions to water, wastes to treatment, and output products.
        - Extract all parameters related to each input and output category from the given context. Ensure completeness and precision in your extraction. 
        - If the input or output has multiple distinct values, each value should be represented by a separate input/output entity.
        - The value of the input and output must be numeric and accompanied by a unit, and cannot be a string or text
        - Avoid extra explanations and format your response as follows:
            - "Inputs": [] # list format 
            - "Outputs": [] # list format 
        - If you do not find any information about the input and outp, label them as "Not mentioned."

        ### Steps of data Extraction: 
        1. Understand the format of the markdown table, including the meaning of each column and row.
        2. Based on the provided pathway and activity, extract all relevant inputs and outputs within the table.

        ###Examples:
        1. Question - Extract inputs and outputs of activity “Hydrogen generation” of pathway “PV/CCU-CH 3 OH technical route” from the Table: <The inventory data for PV/CCU-CH 3 OH technical route\n\nInputs Values Units Outputs Values Units\n\n**Hydrogen generation**\n
        Electricity 10070 kWh Hydrogen 190 kg\n\nWater 1710 MJ Waste water 87.4 kg\n\nPotassium hydroxide 0.57 kg Oxygen 1501 kg\n\nHydrogen plant 1.02E-6 Unit\n\n**Methanol synthesis and purification**\nElectricity 42.75 kWh Methanol 1000 kg\n\nHydrogen 190 MJ Waste water 630 kg\n\nCarbon dioxide (HP) 1455 kg\n\n
        Aluminium oxide 0.012 kg\n\nCopper oxide 0.062 kg\n\nZinc oxide 0.029 kg>
        Answer: 
        - “Inputs”: [“Electricity”: 10070 kWh, “Water”: 1710 MJ, “Potassium”: 0.57 kg, “Hydrogen plant”: 1.02E-6 Unit]
        - “Outputs”: [“Hydrogen”: 190 kg, “Waste water”: 87.4 kg, “Oxygen”: 1501 kg]
        2. Question - Extract inputs and outputs of activity “coal to EG” of pathway “CtEG route” from the Table: <Table. Life Cycle Energy Consumption Inventory Data of the CtEG Route for Producing 1 ton EG\n\nparameter unit coal mining and processing coal transportation coal to EG total life cycle\n\nMaterials Consumption [8]\n\n
        coal t 3.17 × 10 [0] 3.17 × 10 [0]\n\nEnergy Consumption [8] [,] [22] [,] [31]\n\nfuel coal GJ 2.89 × 10 [0] 4.73 × 10 [1] 5.02 × 10 [1]\n\ndiesel GJ 5.79 × 10 [−] [2] 2.47 × 10 [−] [1] 3.05 × 10 [−] [1]\n\ngasoline GJ 5.79 × 10 [−] [2] 1.73 × 10 [−] [2] 7.52 × 10 [−] [2]\n\nelectricity GJ 5.79 × 10 [−] [1] 1.47 × 10 [−] [1] 3.16 × 10 [0] 3.89 × 10 [0]\n\n
        Pollutant Emission [8] [,] [22] [,] [32]\n\nCO 2 kg 1.26 × 10 [2] 6.30 × 10 [1] 6.72 × 10 [3] 6.91 × 10 [3]\n\nCH 4 kg 2.37 × 10 [0] 3.42 × 10 [−] [1] 2.44 × 10 [1] 2.71 × 10 [1]\n\nN 2 O kg 2.13 × 10 [−] [3] 1.47 × 10 [−] [4] 7.69 × 10 [−] [3] 9.97 × 10 [−] [3]\n\nCO kg 1.51 × 10 [−] [2] 5.41 × 10 [−] [1] 1.06 × 10 [1] 1.12 × 10 [1]\n\nNO x kg 4.67 × 10 [−] [1] 1.17 × 10 [−] [1] 3.53 × 10 [0] 4.11 × 10 [0]\n\n
        SO 2 kg 1.69 × 10 [−] [1] 2.69 × 10 [−] [2] 1.44 × 10 [0] 1.64 × 10 [0]\n\nPM 10 kg 5.23 × 10 [−] [1] 1.08 × 10 [−] [1] 2.92 × 10 [0] 3.55 × 10 [0]\n\nVOC kg 8.24 × 10 [−] [3] 8.00 × 10 [−] [3] 8.58 × 10 [−] [2] 1.02 × 10 [−] [1]\n\n>
        - “Inputs”: [“coal”: 3.17 t, “fuel coal”: 47.3 GJ, “electricity”: 3.16 GJ]
        - “Outputs”: [“CO2”: 6720 kg, “CH4”: 24.4 kg, “N2O”: 0.00769 kg, “CO”: 10.6 kg, “NOx”: 3.53 kg, “SO2”: 1.44 kg, “PM10”: 2.92 kg, “VOC”: 0.0858 kg]
        3. Question - Extract inputs and outputs of activity “” of pathway “Petro-EG” from the Table: <Table.** Inventory to produce 1 metric ton of EG via different pathways[14, 18]\n\nUnit Petro-EG Coal-EG\n\nFeedstock\n\nCoal t 3.17\n\nMethanol t 0.60\n\nNH 3 t 0.30\n\nEthylene t 0.75\n\nWater t 3\n\nUtilites\n\nElectricity kWh 300 1200\n\n4.2 MPa steam t 0.10 0.42\n\n
        1.5 MPa steam t 0.40 4.50\n\n0.5 MPa steam t 0.10 3.90\n\nDirect CO 2 emissions t 0.94 5.44\n\n17\n>
        Answer:
        - “Inputs”: [“Ethylene”: 0.75 t, “Water”: 3 t, “Electricity”: 300 kWh, “4.2 MPa steam”: 0.1 t, “1.5 MPa steam”: 0.4 t, “0.5 MPa steam”: 0.1 t]
        - “Outputs”: [“Direct CO2 emissions”: 0.94 t]


        {chat_history}
        {context}
        Question: {question}
        """

    prompt2 = PromptTemplate(
        template=template2, input_variables=["context", "chat_history", "question"], output_parser=CommaSeparatedListOutputParser()
    )

    #set up chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt2}
    )

    #get the result
    result2 = []

    query2 = f"Extract inputs and outputs of the activity {activity} from the pathway {pathway} from the [{table_title}]."
    result = chain({"question": query2})
    result2.append(result["answer"].strip('```json').strip('```').strip())
    
    msgs.clear()
    memory.clear()
    
    return result2

def convert_to_dataframe (raw_text, openai_api_key):
    """Convert results into the structured dataframe """
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=4000,
        openai_api_key=openai_api_key
    )
    
    template3 = """
    You are a helpful assistant specializing in converting life cycle assessment (LCA) data format. Based on the provided context, use the definitions below to answer the question at the end.

    ### Convert the provided list into a structured format based on the definitions below: 
    - **Flow name**: the name of the flow.
    - **Input/Outputs**: specifies whether it is an input or an output flow.
    - **Amount**: the quantity of the flow.
    - **Unit**: the unit in which the amount is measured.
    
    ###Avoid extra explanations and format your response as JSON format following:[{{"Flow name": "", "Input/Output": "", "Amount": "", "Unit": ""}}, {{"Flow name": "", "Input/Output": "", "Amount": "", "Unit": ""}}]

    ###Examples:
    1. Question: <['- "Inputs": ["Electricity": 10070 kWh, "Water": 1710 MJ, "Potassium hydroxide": 0.57 kg, "Hydrogen plant": 1.02E-6 Unit]\n- "Outputs": ["Hydrogen": 190 kg, "Waste water": 87.4 kg, "Oxygen": 1501 kg]']>
    Answer: 
    [
    {{"Flow name": "Electricity", "Input/Output": "Input", "Amount": 10070, "Unit": "kWh"}},
    {{"Flow name": "Water", "Input/Output": "Input", "Amount": 1710, "Unit": "MJ"}},
    {{"Flow name": "Potassium hydroxide", "Input/Output": "Input", "Amount": 0.57, "Unit": "kg"}},
    {{"Flow name": "Hydrogen plant", "Input/Output": "Input", "Amount": 1.02e-6, "Unit": "Unit"}},
    {{"Flow name": "Hydrogen", "Input/Output": "Output", "Amount": 190, "Unit": "kg"}},
    {{"Flow name": "Waste water", "Input/Output": "Output", "Amount": 87.4, "Unit": "kg"}},
    {{"Flow name": "Oxygen", "Input/Output": "Output", "Amount": 1501, "Unit": "kg"}}
    ]
    2. Question: <['- "Inputs": ["coal": 3.17 t, "fuel coal": 50.2 GJ, "diesel": 0.0305 GJ, "gasoline": 0.00752 GJ, "electricity": 3.16 GJ]\n- "Outputs": ["CO2": 6720 kg, "CH4": 24.4 kg, "N2O": 0.00769 kg, "CO": 10.6 kg, "NOx": 3.53 kg, "SO2": 1.44 kg, "PM10": 2.92 kg, "VOC": 0.0858 kg]']>
    Answer: 
    [
    {{"Flow name": "Coal", "Input/Output": "Input", "Amount": 3.17, "Unit": "t"}},
    {{"Flow name": "Fuel Coal", "Input/Output": "Input", "Amount": 50.2, "Unit": "GJ"}},
    {{"Flow name": "Diesel", "Input/Output": "Input", "Amount": 0.0305, "Unit": "GJ"}},
    {{"Flow name": "Gasoline", "Input/Output": "Input", "Amount": 0.00752, "Unit": "GJ"}},
    {{"Flow name": "Electricity", "Input/Output": "Input", "Amount": 3.16, "Unit": "GJ"}},
    {{"Flow name": "CO2", "Input/Output": "Output", "Amount": 6720, "Unit": "kg"}},
    {{"Flow name": "CH4", "Input/Output": "Output", "Amount": 24.4, "Unit": "kg"}},
    {{"Flow name": "N2O", "Input/Output": "Output", "Amount": 0.00769, "Unit": "kg"}},
    {{"Flow name": "CO", "Input/Output": "Output", "Amount": 10.6, "Unit": "kg"}},
    {{"Flow name": "NOx", "Input/Output": "Output", "Amount": 3.53, "Unit": "kg"}},
    {{"Flow name": "SO2", "Input/Output": "Output", "Amount": 1.44, "Unit": "kg"}},
    {{"Flow name": "PM10", "Input/Output": "Output", "Amount": 2.92, "Unit": "kg"}},
    {{"Flow name": "VOC", "Input/Output": "Output", "Amount": 0.0858, "Unit": "kg"}}
    ]
    3. Question: <['- "Inputs": ["Coal": 3.17 t, "Methanol": 0.60 t, "NH3": 0.30 t, "Ethylene": 0.75 t, "Water": 3 t, "Electricity": 300 kWh, "4.2 MPa steam": 0.10 t, "1.5 MPa steam": 0.40 t, "0.5 MPa steam": 0.10 t]\n- "Outputs": ["Direct CO2 emissions": 0.94 t]']>
    Answer: 
    [
    {{"Flow name": "Coal", "Input/Output": "Input", "Amount": 3.17, "Unit": "t"}},
    {{"Flow name": "Methanol", "Input/Output": "Input", "Amount": 0.60, "Unit": "t"}},
    {{"Flow name": "NH3", "Input/Output": "Input", "Amount": 0.30, "Unit": "t"}},
    {{"Flow name": "Ethylene", "Input/Output": "Input", "Amount": 0.75, "Unit": "t"}},
    {{"Flow name": "Water", "Input/Output": "Input", "Amount": 3, "Unit": "t"}},
    {{"Flow name": "Electricity", "Input/Output": "Input", "Amount": 300, "Unit": "kWh"}},
    {{"Flow name": "4.2 MPa steam", "Input/Output": "Input", "Amount": 0.10, "Unit": "t"}},
    {{"Flow name": "1.5 MPa steam", "Input/Output": "Input", "Amount": 0.40, "Unit": "t"}},
    {{"Flow name": "0.5 MPa steam", "Input/Output": "Input", "Amount": 0.10, "Unit": "t"}},
    {{"Flow name": "Direct CO2 emissions", "Input/Output": "Output", "Amount": 0.94, "Unit": "t"}}
    ]
    """
        
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", template3),
        ("human", "Respond to question: {question}")
    ])

    # Insert a question into the template and call the model
    result3 = []
    
    query3 = f"Convert the list {raw_text} into defined data format"
    full_prompt = prompt_template.format_messages(question=query3)
    result = llm.invoke(full_prompt)
    result3.append(result)

    data = result3[0].content
    
    return data

def data_to_dataframe(data):
    # Step 1: Remove starting and ending triple backticks
    data = data.strip()
    if data.startswith('```json'):
        data = data[7:]  # Remove ```json\n
    if data.startswith('```'):
        data = data[3:]  # If somehow only ```
    if data.endswith('```'):
        data = data[:-3]
    data = data.strip()

    # Step 2: Directly load the JSON list
    data_list = json.loads(data)

    # Step 3: Convert to DataFrame
    df = pd.DataFrame(data_list)
    return df

def process_document_LCI (df_table_title, df_vectordb):
    
    df_main = pd.merge(df_table_title, df_vectordb, on='PDF_path', how='inner')
    df_main = df_main.drop(["Num_pages", "Num_chunks", "Is_added_to_vectorDB"], axis=1)

    df_grouped = {table: df_group for table, df_group in df_main.groupby("Paper_title", sort=False)}

    return df_grouped

def process_table(df_paper, openai_api_key):
    table_info = []

    for _, row in df_paper.iterrows():
        table_title = row['Table_title']
        persist_directory = row['Vectordb_path']
        page_number = row['Page']

        data_dict, retriever = LCI_pathway_extraction(table_title, page_number, persist_directory, openai_api_key)

        pathways = data_dict.get("Pathway", [])
        activities = data_dict.get("Activity", [])

        # Ensure both are lists
        if isinstance(pathways, str):
            pathways = [pathways]
        if isinstance(activities, str):
            activities = [activities]

        if not pathways or not activities:
            continue  # Skip if missing

        pathway = pathways[0]

        rows = [(pathway, activity) for activity in activities]

        df_table = pd.DataFrame(rows, columns=["Pathway", "Activity"])
        df_table["Paper_title"] = row["Paper_title"]
        df_table["Table_title"] = table_title
        df_table["Vectordb_path"] = persist_directory
        df_table["Page"] = page_number

        table_info.append(df_table)

    df_table = pd.concat(table_info, ignore_index=True)

    return df_table, retriever

def extract_LCI(df_table, retriever, openai_api_key):
    LCI_db = {}  # Dictionary to store extracted LCI data

    for _, row in df_table.iterrows():
        table_title = row["Table_title"]
        pathway = row["Pathway"]
        activity = row["Activity"]
        persist_directory = row['Vectordb_path']
        page_number = row['Page']

        print(f"Processing Table: {table_title}, Pathway: {pathway}, Activity: {activity}")

        # Extract raw LCI data
        raw_data = LCI_data_extraction(table_title, page_number, pathway, activity, persist_directory, openai_api_key)

        # Convert raw data into a DataFrame
        data = convert_to_dataframe(raw_data, openai_api_key)
        df_LCI = data_to_dataframe(data)

        # Add metadata to LCI data
        df_LCI["Pathway"] = pathway
        df_LCI["Activity"] = activity

        # Append data to the existing table title if already present
        if table_title in LCI_db:
            LCI_db[table_title] = pd.concat([LCI_db[table_title], df_LCI], ignore_index=True)
        else:
            LCI_db[table_title] = df_LCI

    return LCI_db

def save_LCI_db_to_csv(LCI_db, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        for table_title, df in LCI_db.items():
            f.write(f"# {table_title}\n")  # Add a header for each table
            df = df.dropna(how='all')  # Remove empty rows
            df.to_csv(f, index=False)  # Write DataFrame without extra blank lines

def classify_LCI_inventory_tables(pdf_directory, model_path, openai_api_key):

    # Step 1: Load PDF file paths
    pdf_files = load_pdf_paths_table(pdf_directory)

    # Step 2: Extract table titles and metadata
    df1 = [extract_tables_and_pages(path) for path in pdf_files]
    df1 = pd.concat(df1, ignore_index=True)

    # Step 3: Generate embeddings for table titles
    text_embedding_list = generate_embeddings(df1['Table_title'], openai_api_key)
    text_embedding_list = [e.embedding for e in text_embedding_list]
    embeddings_array = np.array(text_embedding_list)

    # Step 4: Load classifier and predict
    clf_loaded = load(model_path)
    predictions = clf_loaded.predict(embeddings_array)

    # Step 5: Combine with original data
    df2 = pd.DataFrame(predictions)
    combined_df = pd.concat([df1, df2], axis=1).fillna(False)
    combined_df.rename(columns={0: 'is LCI inventory table?'}, inplace=True)

    # Step 6: Filter predicted LCI inventory tables
    df_filtered = combined_df[combined_df["is LCI inventory table?"] == True]

    return df_filtered

def process_all_LCI_outputs_by_id(df_grouped, output_folder, openai_api_key):
    """Process LCI_db for each paper and save to individual CSVs using index as file name"""

    saved_paths = []

    for idx, (paper_title, df_paper) in enumerate(df_grouped.items()):
        try:
            # Process table and extract LCI
            df_table, retriever = process_table(df_paper, openai_api_key)
            LCI_db = extract_LCI(df_table, retriever, openai_api_key)

            # File name as index (e.g., "0.csv", "1.csv", ...)
            file_name = f"{idx+1}.csv"
            file_path = os.path.join(output_folder, file_name)

            # Save LCI_db to CSV
            save_LCI_db_to_csv(LCI_db, file_path)

            # Track metadata
            saved_paths.append({
                "ID": idx+1,
                "Paper_title": paper_title,
                "File_path": file_path
            })

            print(f"✅ Saved {file_path}")

        except Exception as e:
            print(f"❌ Error processing {paper_title}: {e}")

    return pd.DataFrame(saved_paths)

st.title("LCA Domain Data Extraction")

st.subheader("1. Build VectorDB")  

# Input fields
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
pdf_directory = st.text_input("Enter path to PDF folder:")
base_persist_directory = st.text_input("Enter path for Vector DB storage:")

if openai_api_key and pdf_directory and base_persist_directory:
    st.success("✅ All required inputs provided successfully.")

# Then validate and run
if st.button("▶️ Build VectorDB"):
    df_path = load_pdf_paths(pdf_directory, base_persist_directory)
    df_vectordb = build_vectorDB(df_path, openai_api_key)
    st.session_state["df_vectordb"] = df_vectordb
    st.success("✅ Vector DB built successfully.")

# Always display df_vectordb if it exists in session_state
if "df_vectordb" in st.session_state:
    st.dataframe(st.session_state["df_vectordb"])

st.subheader("2. Extract LCIA results & LCA modeling assumptions data") 

if st.button("▶️ Extract LCIA results & LCA modeling assumptions"):
    if "df_vectordb" in st.session_state:
        with st.spinner("🔍 Extracting data from documents..."):
            try:
                all_results = process_document(st.session_state["df_vectordb"], openai_api_key)
                df1 = pd.DataFrame(all_results)
                st.session_state["df1"] = df1
                st.success("✅ Extraction completed successfully.")
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")

# Display df1 if exists
if "df1" in st.session_state:
    st.dataframe(st.session_state["df1"])
    csv = st.session_state["df1"].to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="LCA_extraction_results.csv", mime="text/csv")

st.subheader("3. Extract LCI data")

rf_model_path = st.text_input("Enter path for random forest classifer:")
output_folder = st.text_input("Enter path for LCI data storage:")

if rf_model_path and output_folder:
    st.success("✅ All required inputs provided successfully.")

if st.button("▶️ Extract LCI inventory data"):
    if "df_vectordb" in st.session_state:
        with st.spinner("🔍 Extracting LCI inventory data..."):
            try:
                df_filtered= classify_LCI_inventory_tables(pdf_directory, rf_model_path, openai_api_key)
                st.session_state["df_filtered"] = df_filtered
                df_vectordb = st.session_state["df_vectordb"]
                df_grouped = process_document_LCI(df_filtered, df_vectordb)

                os.makedirs(output_folder, exist_ok=True)
                df_saved_paths = process_all_LCI_outputs_by_id(df_grouped, output_folder, openai_api_key)

                st.session_state["df_saved_paths"] = df_saved_paths
                st.success("✅ Extraction completed successfully.")

            except Exception as e:
                st.error(f"❌ LCI extraction failed: {e}")

if "df_saved_paths" in st.session_state:
    st.dataframe(st.session_state["df_saved_paths"])
    csv = st.session_state["df_saved_paths"].to_csv(index=False)
    st.download_button("Download LCI File Paths", data=csv, file_name="LCI_file_paths.csv", mime="text/csv")
