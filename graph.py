from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

load_dotenv(override=True)

texts = ["""
Je m'appelle Alaaeddine, je suis étudiant en ingénierie informatique à l’EMSI.

Je suis passionné par le développement web et mobile, ainsi que par la conception de systèmes intelligents.

Je travaille principalement avec des technologies comme React, React Native, Laravel, MongoDB et Django.

J’ai développé un projet de Smart Traffic System visant à améliorer la gestion du trafic routier à l’aide de technologies intelligentes et de traitement de données en temps réel.

J’ai également développé une application mobile appelée TuneDive, une application inspirée de Spotify permettant aux utilisateurs d’écouter et découvrir de la musique avec une expérience personnalisée.

Dans mes projets, je travaille sur des architectures full-stack incluant la création d’APIs, la gestion de bases de données et le développement d’interfaces utilisateur modernes.

Je m’intéresse aux systèmes distribués, aux applications mobiles cross-platform et à l’optimisation des expériences utilisateur.

Je continue à améliorer mes compétences à travers des projets concrets combinant développement logiciel, logique système et design d’applications.
"""]

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts= texts,
    collection_name="cv_collection",
    embedding=embedding_model

)
retriever= vectorstore.as_retriever()

retriever_tool = create_retriever_tool(

    retriever=retriever,
    name="cv_tool",
    description="get information about me ",

)    

@tool
def get_employee_info(name: str) :
    """
    Get information about a given employee (name, salary, seniority)
    """
    print("get_employee_info tool invoked")
   
    return {"name": name, "salary": "12000", "seniority": "5"}
@tool
def send_email(email:str, subject:str, content:str):
    """
    Send an email with subject and content
    """
    
    print(f"sending email to {email}, subject :{subject},content :{content}")
    return f"Email succesfully sent to {email} with subject {subject} and content {content}"

llm=ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email,retriever_tool],
    system_prompt="answer to user query using provided tools"
)
resp=agent.invoke(
 input={"messages": [HumanMessage("c'est quoi le salaire de John Doe ?")]})
print(resp['messages'][-1].content)