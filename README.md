# LCAKG_RAG

This is the repository for our upcoming publication:  

Tang, Z., Dreger, M., Jiang, P., Malek, K., & Tu, Q. (2026) From Literature to Knowledge Graphs: Automated Extraction and Retrieval of Life Cycle Assessment Data with Large Language Models.

## **Project Overview**
The large and steadily growing volume of scientific literature poses challenges for accessing and utilizing data due to its unstructured nature. Life cycle assessment (LCA), in particular, relies on high-quality life cycle inventory (LCI) data to quantify the environmental impacts of products or services across their life cycle, yet manual extraction of such data is time-consuming and labor-intensive. This study presents an automated framework that integrates large language models (LLMs) with knowledge graph (KG) to extract and manage LCA data from published studies. A retrieval-augmented generation (RAG) pipeline is developed to mine three core data types from full-text articles: LCI data, life cycle impact assessment (LCIA) results, and LCA modeling assumptions. The extracted information is normalized and mapped to an ontology-driven, LCA-oriented knowledge graph (LCAKG) implemented in Neo4j. To support user interaction, an LLM-based question-answering system translates natural language queries into executable graph queries, allowing  users to retrieve rich information without prior knowledge of KG schemas. The framework is evaluated using a case study of LCA studies on chemical production. The results demonstrate high semantic accuracy in data extraction, with F1-scores ranging from 73.54% to 93.34%. Query performance is significantly improved by combining similarity search with text-to-cypher reasoning, increasing the F1-score from 56.98% (baseline) to 75.18%. The proposed framework enhances the accessibility and interoperability of LCA domain data and provides a scalable foundation for large-scale knowledge synthesis to support LCA research.

### **Framework ovewview:** 
The model architecture and workflow are illustrated below:
Add picture here

![LCAKG framework](figures/LCAKG_framework.png)

## **Features**

The main features of this pipeline:

* Extraction of table titles and their corresponding page numbers from PDFs.

* Text Embedding classification of table titles to identify LCI inventory tables.

* Extraction of LCI inventory data, LCA modeling assumptions, and LCIA results from LCA studies.

* Construction of knowledge graphs based on the LCAKG ontology.

* Retrieval of information using Neo4j Cypher or the Q & A system.


**Installation**  

To install and run the app locally, follow these steps:

1. Clone this repository to your local machine.

2. Install the required dependencies using the provided requirements.txt file for each folder.

3. Run the code.

