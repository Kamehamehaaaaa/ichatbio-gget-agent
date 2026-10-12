import json
import os
from typing import Literal

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from context import current_context
from tools.util import context_tool
from util import get_llm_credentials

# Assumes each file in models/ defines a class with the same name as the file.
from models.AlignSequences import AlignSequences
from models.BlastInput import BlastInput
from models.ViralSequence import ViralSequence

# The LLM picks a task name (plain string); the tool maps it to the right schema.
TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "species_identification": BlastInput,
    "viral_sequence": ViralSequence,
    "align_sequences": AlignSequences,
}

SYSTEM_PROMPT = (
    "You extract parameters for gget commands that retrieve and analyze nucleotide/protein data "
    "from NCBI. Fill in only what the user's request states or clearly implies. Copy sequences "
    "and accession numbers exactly. Leave fields empty rather than guessing."
)


@context_tool
async def extract_parameters(
    user_query: str,
    task: Literal["species_identification", "viral_sequence", "align_sequences"],
) -> str:
    """
    Extracts structured gget parameters from the user's request. Call this first.

    :param user_query: The user's request, verbatim.
    :param task: Which operation the user wants:
        - "species_identification": identify the organism a sequence comes from (BLAST)
        - "viral_sequence": retrieve viral sequence records
        - "align_sequences": align two or more sequences
    :return: JSON with the task and the extracted parameters, to pass to the next tool.
    """
    context = current_context.get()
    schema = TASK_SCHEMAS[task]

    async with context.begin_process("Extracting parameters from request") as process:
        creds = get_llm_credentials()
        client = instructor.from_openai(
            AsyncOpenAI(api_key=creds["api_key"], base_url=creds["base_url"])
        )

        try:
            params = await client.chat.completions.create(
                model=os.getenv("LLM"),
                response_model=schema,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                temperature=0,
                max_retries=2,
            )
        except Exception as e:
            await process.log(f"Parameter extraction failed: {e}")
            return f"ERROR: could not extract parameters for task '{task}': {e}"

        generation = params.model_dump(exclude_none=True, by_alias=True)
        await process.log(f"Extracted parameters: {generation}")

    # This return value is what the LLM sees and uses to call the next tool.
    return json.dumps({"task": task, "parameters": generation})




# from pydantic import BaseModel
# from typing import Literal, Optional, Type
# from openai import AsyncOpenAI
# import instructor
# from tools.util import context_tool

# from util import get_llm_credentials, set_llm_credentials

# from context import current_request, current_context, current_artifacts
# import json

# @context_tool
# async def extract_parameters(
#     user_query: str, 
#     schema: Type[BaseModel]) -> BaseModel:

#     """
#     extract_parameters is a tool that extracts parameters from a user query based on a provided Pydantic schema. 
#     It uses an LLM to parse the user query and return a structured response that adheres to the schema.
#     """

#     context = current_context.get()

#     async with context.begin_process("Extracting parameters from request") as process:

#         system_prompt = f'You are an parameter extractor for the gget command used to get nucleotide data to extract data relevant to user query.'

#         response_model = schema

#         api_key, base_url = get_llm_credentials().values()
            
#         client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
#         instructor_client = instructor.patch(client)

#         req = await instructor_client.chat.completions.create(
#             model="gpt-oss-120b",
#             response_model=response_model,
#             messages=[
#                 {"role": "system",
#                     "content": system_prompt},
#                 {"role": "user", "content": user_query}],
#             temperature=0,
#         )

#         generation = req.model_dump(exclude_none=True, by_alias=True)

#         await process.log(f"Extracted parameters: {generation}")

#         await process.create_artifact(
#             mimetype="application/json",
#             description=f"Extracted parameters from user query",
#             content=json.dumps(generation).encode("utf-8"),
#             metadata={
#                 "source_artifacts": []
#             },
#         )