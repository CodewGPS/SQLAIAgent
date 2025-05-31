from phi.agent import Agent
from phi.tools.sql import SQLTools
from phi.model.groq import Groq
from phi.model.ollama import Ollama
import chainlit as cl
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

def create_agent(db_url: str):
    sql_agent = Agent(
        tools=[SQLTools(db_url=db_url)],
        model=Groq(id="llama3-70b-8192", api_key=api_key),
        #model=Ollama(id="gemma3:4b"),
        add_chat_history_to_messages=True,
        num_history_responses=3,
        description="You are a helpful AI agent that answers questions about the SQL database and the tables in it. Your responses must be detailed and accurate. Do not respond to any out-of-context questions.",
        instructions=[
            "Answer the questions related to the SQL database in detail.",
            "Do not answer any questions which are out of the context of the database.",
            "If the user greets you, greet them back politely.",
            "Include the SQL Query used to get the answer in your reply."
        ]
    )
    return sql_agent


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 Hello! Please enter your database connection URL to get started.").send()
    cl.user_session.set("awaiting_db_url", True)


@cl.on_message
async def on_message(message: cl.Message):
    try:
        if cl.user_session.get("awaiting_db_url"):
            db_url = message.content.strip()
            cl.user_session.set("db_url", db_url)
            sql_agent = create_agent(db_url)
            cl.user_session.set("agent", sql_agent)
            cl.user_session.set("awaiting_db_url", False)
            await cl.Message(content="✅ Database URL received. You can now start asking questions about your database.").send()
            return

        agent = cl.user_session.get("agent")
        cl.chat_context.to_openai()
        msg = cl.Message(content="")
        for chunk in await cl.make_async(agent.run)(message.content, stream=True):
            await msg.stream_token(chunk.get_content_as_string())
        await msg.send()

    except KeyError as e:
        await cl.Message(content=f"❌ Error: Missing key in session: {e}").send()

    except AttributeError as e:
        await cl.Message(content=f"❌ Error: Invalid operation: {e}").send()

    except Exception as e:
        await cl.Message(content=f"❌ An unexpected error occurred: {e}").send()
