

# Medical-Chatbot-with-LLMs-LangChain-Pinecone-and-Flask
This project is a Flask-based medical chatbot that leverages a Retrieval-Augmented Generation (RAG) pipeline. The medical PDFs are stored in the data folder, where they are split into chunks and embedded using the sentence-transformers/all-MiniLM-L6-v2 model. These embeddings are stored in a Pinecone index called medical-chatbot for fast semantic retrieval. During query time, LangChain retrieves the most relevant chunks from Pinecone and combines them with a medical assistant system prompt defined in the prompt.py file. The combined input is then passed to the Groq LLM (ChatGroq, Llama-3.1), which generates concise and source-grounded answers that are displayed through the web interface at the home page


# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/sarth360/Medical-Chatbot-with-LLMs-LangChain-Pinecone-and-Flask.git

```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.12 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Groq
- Pinecone



