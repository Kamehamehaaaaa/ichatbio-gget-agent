import dotenv
import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

dotenv.load_dotenv()

# See https://langchain-ai.github.io/langgraph/agents/overview/#visualize-an-agent-graph for a nice visualization
# of the react agent graph


@pytest.mark.skip(reason="Just for reference")
@pytest.mark.asyncio
async def test_haha():
    aborted = False

    @tool(return_direct=True)
    async def abort(reason: str):
        """If you can't do what was asked, abort instead and explain why."""
        nonlocal aborted
        aborted = True

    llm = ChatOpenAI(model="gpt-4.1-mini", tool_choice="required")

    agent = create_agent(
        model=llm,
        tools=[abort],
        prompt="You are a helpful assistant, but you can't do much, so just abort.",
    )

    # Run the agent
    try:
        out = await agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
    except GraphRecursionError as e:
        pass
    pass

    assert aborted
