from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter():

    return RecursiveCharacterTextSplitter(
       
        separators=[
       
        "\n\n",
       
        "\n",
        
        ". ",
        
        " ",
        
        ""
        ],

        chunk_size=500,

        chunk_overlap=100)
    