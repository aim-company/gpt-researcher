from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


class AgentDescription(BaseModel):
    """
    Represents the server and agent role prompt for a specific task.

    Attributes:
        server (str): An emoji and title indicating the type of agent, such as "💰 Finance Agent".
        agent_role_prompt (str): A detailed prompt that defines the role and objectives of the agent,
                                 such as composing comprehensive, astute, impartial, and methodically arranged reports.
    """

    server: str
    agent_role_prompt: str

    @staticmethod
    def get_parser():
        return PydanticOutputParser(pydantic_object=AgentDescription)

    @staticmethod
    def get_instructions():
        return AgentDescription.get_parser().get_format_instructions()
