"""
This agent runs as a LangChain ReAct-style agent with enforced tool-calling. Such agents operate by iteratively calling
tools in order to fulfill the user's request. This is modeled as a conversation which begins with just the user's
request, then each subsequent toolcall appends agent-generated messages to the conversation. This particular agent has
access to two special tools - "abort" and "finish" - which the agent calls when it decides that either it has
successfully fulfilled the user's request ("finish") or that it isn't able to do so and should quit instead ("abort").

The agent's LLM both picks which tool to call and fills in that tool's arguments from the user's request, so tools
themselves need no LLM calls. Each tool returns a short result for the LLM and saves full outputs as iChatBio artifacts.

See the flowchart in README.md for a visualization of the agent.
"""

import json
import logging
import os
from typing import Any, override

import dotenv
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from context import current_context, current_request
from tools.get_sequence import get_sequence
from tools.run_blast import run_blast_sync
from tools.get_virus import get_virus
from tools.ensembl_search import ensembl_search
from tools.gene_info import gene_info
from tools.get_ensembl_sequence import get_ensembl_sequence
from tools.reference_genome import reference_genome
from util import get_llm_credentials, set_llm_credentials

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gget_agent")

# Max graph steps per request; stops runaway tool loops
RECURSION_LIMIT = 20

DESCRIPTION = """\
Useful for querying NCBI's nuccore nucleotide database and identifying species from sequences.
This agent can do the following:
- Retrieve a nucleotide sequence record from NCBI given an accession number (e.g. NM_000546.6, MN908947).
- Identify the likely source organism(s) of a nucleotide or protein sequence by BLASTing it against
  NCBI databases via gget. Accepts a raw sequence pasted by the user or an NCBI accession number,
  with optional BLAST settings (program, database, number of hits, e-value).
- Download virus genome sequences and metadata from NCBI Virus, filtered by host, location, collection date, completeness, lineage and more.
- Search Ensembl for genes by name or keyword in any species and return their Ensembl IDs.
- Look up Ensembl gene/transcript details: location, biotype, canonical transcript, UniProt and NCBI Gene IDs, function.
- Fetch nucleotide or protein sequences for Ensembl gene/transcript IDs.
- Get download links for Ensembl reference genomes and annotations (FASTA, GTF) for any species.
"""


class EntrypointParameters(BaseModel):
    user_request: str = Field(description="The user's request", min_length=1)
    sequence: str | None = Field(
        None,
        description=(
            "The raw nucleotide or protein sequence the user provided, copied exactly and in full "
            "(FASTA headers allowed). Required whenever the request refers to a pasted sequence."
        ),
    )


class GgetNucleotideAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Gget Nucleotide Agent",
            description=DESCRIPTION,
            icon="https://raw.githubusercontent.com/acislab/ichatbio/refs/heads/main/icon.svg",
            documentation_url="https://github.com/acislab/ichatbio-data-handler-agent",
            url=os.getenv("URL", None),
            entrypoints=[
                AgentEntrypoint(
                    id="process_request",
                    description=(
                        "Processes a user request to retrieve nucleotide sequences from NCBI "
                        "and identify species from sequences via BLAST."
                    ),
                    parameters=EntrypointParameters,
                )
            ],
        )

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: EntrypointParameters,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Builds a LangChain agent graph (a loop that alternates between decision-making and tool execution), then
        executes it with `request` as input. Tools send messages back to iChatBio via the `context` object, which is
        stored in a context var so concurrent requests stay isolated.
        """
        set_llm_credentials(metadata)

        current_request.set(request)
        current_context.set(context)

        # Terminal tools

        @tool(return_direct=True)  # This tool ends the agent loop
        async def abort(reason: str):
            """If you can't fulfill the user's request, abort instead and explain why."""
            await context.reply(reason)

        @tool(return_direct=True)  # This tool ends the agent loop
        async def finish(message: str):
            """Mark the user's request as successfully completed and give the user a summary of the results."""
            await context.reply(message)

        tools = [
            get_sequence,
            run_blast_sync,
            get_virus,
            ensembl_search,
            gene_info,
            get_ensembl_sequence,
            reference_genome,
            abort,
            finish,
        ]

        # Build the LangChain agent graph

        llm_credentials = get_llm_credentials()
        llm = ChatOpenAI(
            model=os.getenv("LLM"),
            tool_choice="required",
            openai_api_key=llm_credentials["api_key"],
            openai_api_base=llm_credentials["base_url"],
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_MESSAGE,
        )

        # Run the graph, logging each tool call and result
        user_message = request
        if params.sequence:
            user_message += f"\n\nSequence provided by the user:\n{params.sequence}"


        logger.info("Request: %s", user_message)

        async for update in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="updates",
            config={"recursion_limit": RECURSION_LIMIT},
        ):
            for node, state in update.items():
                for msg in (state or {}).get("messages", []):
                    if getattr(msg, "tool_calls", None):
                        for call in msg.tool_calls:
                            logger.info(
                                "→ TOOL CALL %s(%s)",
                                call["name"],
                                json.dumps(call["args"], default=str)[:500],
                            )
                    elif msg.type == "tool":
                        logger.info("← %s returned: %s", msg.name, str(msg.content)[:500])


SYSTEM_MESSAGE = """\
You are a gget bioinformatics agent. You answer questions about genes and sequences using NCBI and
Ensembl data. You must always respond by calling a tool. When you call a tool, fill in its arguments
from the user's request; copy accession numbers, Ensembl IDs and sequences exactly.

Choosing a database:
- NCBI accessions (NM_, NC_, XM_, MN..., OQ...), BLAST/species identification, and virus datasets -> NCBI tools.
- Ensembl IDs (ENSG..., ENST..., ENSP..., ENSMUSG...), gene names/keywords, reference genomes -> Ensembl tools.

NCBI tools:
- get_sequence: The user wants a sequence or record for an NCBI accession, OR asks which organism or
  species an accession comes from (the record states its organism and taxonomy ID directly).
- run_blast_sync: The user wants to identify the organism of a raw sequence (no accession), or to find
  sequences SIMILAR to a sequence or accession (relatives, homologs). Never use BLAST just to look up
  the organism of an accession. Fill in `settings` only with BLAST options the user explicitly asked for.
Ensembl tools:
- ensembl_search: The user names a gene or keyword (e.g. "ACE2", "insulin receptor") and you need its
  Ensembl ID. Always pass a species (e.g. "homo_sapiens"; default to human only if the user implies it).
- gene_info: The user wants details about Ensembl IDs (location, biotype, transcripts, UniProt/NCBI IDs,
  function). If they gave a gene name, call ensembl_search first and pass the top matching ID.
- get_ensembl_sequence: The user wants the nucleotide or protein sequence of an Ensembl gene/transcript.
  Use translate=true for protein/amino acid sequences. If they gave a gene name, call ensembl_search first.
- reference_genome: The user wants reference genome, transcriptome, protein or GTF annotation files or
  download links for a species. Returns links only.

Ending:
- finish: The request is fulfilled. Summarize the results for the user. For BLAST, start with a
  one-sentence conclusion naming the most likely organism with its % identity and query coverage, then
  list the top hits with % identity, query coverage and e-value. Treat identity >= ~95% over high
  coverage as a species-level match. Never write "low" or "high" in place of a number. Mention that
  full results are saved as artifacts.
- abort: The request is outside these capabilities, or required inputs are missing or invalid.

Typical flows:
1. "Get the sequence for NM_000546.6"               -> get_sequence -> finish
2. "What species is this sequence? ATGC..."         -> run_blast_sync(sequence=...) -> finish
3. "What organism is MN908947 from?"                -> get_sequence(accession="MN908947") -> finish
3b. "Find the 10 closest relatives of MN908947"     -> run_blast_sync(accession="MN908947", settings={"limit": 10}) -> finish
4. "BLAST MN908947 against refseq_rna, top 5 hits"  -> run_blast_sync(accession=..., settings={"database": "refseq_rna", "limit": 5}) -> finish
5. "Complete Zika genomes from humans in Brazil"    -> get_virus(virus="Zika virus", filters={"host": "human", "nuc_completeness": "complete", "geographic_location": "Brazil"}) -> finish
6. "Find human genes related to ACE2"               -> ensembl_search(searchwords=["ace2"], species="homo_sapiens") -> finish
7. "Tell me about ENSG00000130234"                  -> gene_info(ens_ids=["ENSG00000130234"]) -> finish
8. "Protein sequence of human ACE2"                 -> ensembl_search -> get_ensembl_sequence(ens_ids=[top ID], translate=true) -> finish
9. "Human genome FASTA and GTF download links"      -> reference_genome(species="homo_sapiens", which=["dna", "gtf"]) -> finish

When ensembl_search returns several matches, use the one whose gene name exactly matches the user's
gene; if it is ambiguous, list the candidates in finish instead of guessing.

Never invent or abbreviate a sequence (e.g. "ATGC..."). If the request refers to a sequence that is not
actually included, call abort and ask the user to provide the full sequence.

If a tool returns an ERROR, either retry once with corrected arguments or abort and explain the error.
Do not call the same tool repeatedly with the same arguments.
"""

def create_app() -> Starlette:
    dotenv.load_dotenv()

    if os.getenv("LLM") is None:
        raise ValueError("LLM environment variable must be set")

    agent = GgetNucleotideAgent()
    app = build_agent_app(agent)
    return app



# import os
# from typing import Any, override

# import dotenv
# from ichatbio.agent import IChatBioAgent
# from ichatbio.agent_response import ResponseContext
# from ichatbio.server import build_agent_app
# from ichatbio.types import AgentCard, AgentEntrypoint
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_agent
# from pydantic import BaseModel, Field
# from starlette.applications import Starlette

# from context import current_context, current_request
# from tools.extract_parameters import extract_parameters
# from tools.get_sequence import get_sequence
# from tools.run_blast import run_blast_sync
# from util import get_llm_credentials, set_llm_credentials

# import json
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("gget_agent")

# dotenv.load_dotenv()

# DESCRIPTION = """\
# Useful for querying large nucleotide sequence databases like NCBI's nuccore.
# Can search for sequences, retrieve records, and identify species from sequences.
# This agent can do the following:
# - Retrieve nucleotide/protein sequence records from NCBI given an accession number.
# - Identify the likely source organism(s) of a nucleotide or protein sequence by BLASTing it
#   against NCBI databases via gget. Accepts a raw sequence pasted by the user or an NCBI accession number.
# """


# class EntrypointParameters(BaseModel):
#     user_request: str = Field(description="The user's request", min_length=1)


# class GgetNucleotideAgent(IChatBioAgent):
#     @override
#     def get_agent_card(self) -> AgentCard:
#         return AgentCard(
#             name="Gget Nucleotide Agent",
#             description=DESCRIPTION,
#             icon="https://raw.githubusercontent.com/acislab/ichatbio/refs/heads/main/icon.svg",
#             url=os.getenv("URL", None),
#             entrypoints=[
#                 AgentEntrypoint(
#                     id="process_request",
#                     description="Processes a user request to retrieve nucleotide data from NCBI and identify species from sequences.",
#                     parameters=EntrypointParameters,
#                 )
#             ],
#         )

#     @override
#     async def run(
#         self,
#         context: ResponseContext,
#         request: str,
#         entrypoint: str,
#         params: EntrypointParameters,
#         metadata: dict[str, Any] | None = None,
#     ):
#         """
#         Builds a LangChain agent graph (a loop that alternates between decision-making and tool execution), then
#         executes it with `request` as input. The LLM decides which tools to call based on the request and the system
#         prompt. Tools send messages back to iChatBio via the `context` object, which is stored in a context var so
#         concurrent requests stay isolated.
#         """
#         set_llm_credentials(metadata)

#         current_request.set(request)
#         current_context.set(context)

#         # Terminal tools

#         @tool(return_direct=True)  # This tool ends the agent loop
#         async def abort(reason: str):
#             """If you can't fulfill the user's request, abort instead and explain why."""
#             await context.reply(reason)

#         @tool(return_direct=True)  # This tool ends the agent loop
#         async def finish(message: str):
#             """Mark the user's request as successfully completed and give the user a summary of the results."""
#             await context.reply(message)

#         tools = [
#             # extract_parameters,
#             get_sequence,
#             run_blast_sync,
#             abort,
#             finish,
#         ]

#         # Build the LangChain agent graph

#         llm_credentials = get_llm_credentials()
#         llm = ChatOpenAI(
#             model=os.getenv("LLM"),
#             tool_choice="required",
#             openai_api_key=llm_credentials["api_key"],
#             openai_api_base=llm_credentials["base_url"],
#         )

#         agent = create_agent(
#             model=llm,
#             tools=tools,
#             system_prompt=SYSTEM_MESSAGE,
#         )

#         # Run the graph

#         # await agent.ainvoke(
#         #     {
#         #         "messages": [
#         #             {"role": "user", "content": request},
#         #         ]
#         #     }
#         # )
#         async for update in agent.astream(
#             {"messages": [{"role": "user", "content": request}]},
#             stream_mode="updates",
#             config={"recursion_limit": 15},  # stops runaway tool loops
#         ):
#             for node, state in update.items():
#                 for msg in (state or {}).get("messages", []):
#                     if getattr(msg, "tool_calls", None):
#                         for call in msg.tool_calls:
#                             logger.info("→ TOOL CALL %s(%s)", call["name"], json.dumps(call["args"])[:500])
#                     elif msg.type == "tool":
#                         logger.info("← %s returned: %s", msg.name, str(msg.content)[:500])


# SYSTEM_MESSAGE = """\
# You are a gget nucleotide agent. You answer questions about nucleotide and protein sequences
# using NCBI data through the gget Python library. You must always respond by calling a tool.

# Tools and when to use them:
# - extract_parameters: Call this FIRST for every request. It pulls structured inputs out of the
#   user's text (accession numbers, raw sequences, sequence type, BLAST program/database, limits).
# - get_sequence: Use when the user wants a sequence or record for an accession number, OR when the
#   user wants species identification but only gave an accession (fetch the sequence first, then BLAST it).
# - run_blast_sync: Use when the user wants to identify the species/organism of a sequence, or find
#   similar sequences. Needs an actual sequence (pasted by the user or returned by get_sequence).
# - finish: Call when the request is fulfilled. Summarize the results for the user.
# - abort: Call if the request is outside these capabilities, or required inputs are missing or invalid.

# Typical flows:
# 1. "Get the sequence for NC_045512"            -> extract_parameters -> get_sequence -> finish
# 2. "What species is this sequence? ATGC..."     -> extract_parameters -> run_blast_sync -> finish
# 3. "What organism does accession MN908947 come from?"
#                                                -> extract_parameters -> get_sequence -> run_blast_sync -> finish

# Do not call the same tool repeatedly with the same inputs. If a tool fails, either try a reasonable
# fix once or abort with the error explained.
# """


# def create_app() -> Starlette:
#     dotenv.load_dotenv()

#     if os.getenv("LLM") is None:
#         raise ValueError("LLM environment variable must be set")

#     agent = GgetNucleotideAgent()
#     app = build_agent_app(agent)
#     return app






# """
# This agent runs as a LangChain ReAct-style agent with enforced tool-calling. Such agents operate by iteratively calling
# tools in order to fulfill the user's request. This is modeled as a conversation which begins with just the user's
# request, then each subsequent toolcall appends agent-generated messages to the conversation. This particular agent has
# access to two special tools - "abort" and "finish" - which the agent calls when it decides that either it has
# successfully fulfilled the user's request ("finish") or that it isn't able to do so and should quit instead ("abort").

# See the flowchart in README.md for a visualization of the agent.
# """

# import os
# from typing import Any, Iterable, override

# import dotenv
# import langchain.agents
# from ichatbio.agent import IChatBioAgent
# from ichatbio.agent_response import ResponseContext
# from ichatbio.server import build_agent_app
# from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field
# from starlette.applications import Starlette

# from artifact_registry import ArtifactRegistry
# from context import current_artifacts, current_context, current_request
# from tools.extract_parameters import extract_parameters
# from tools.get_sequence import get_sequence
# from tools.run_blast import run_blast_sync
# from util import get_llm_credentials, set_llm_credentials

# dotenv.load_dotenv()

# DESCRIPTION = """\
# Useful for querying large nucleotide sequence databases like NCBI's nuccore. 
# Can search for sequences, retrieve records, and identify species from sequences.
# This agent can do the following:
# - Identifies the likely source organism(s) of a nucleotide or protein sequence by BLASTing it
# against NCBI databases via gget. Accepts either a raw sequence pasted by the user, or a
# reference to an NCBI accession number already known to the agent.


# """


# class EntrypointParameters(BaseModel):
#     user_request: str = Field(
#         description="The user's request", min_length=1
#     )


# class GgetNucleotideAgent(IChatBioAgent):
#     @override
#     def get_agent_card(self) -> AgentCard:
#         return AgentCard(
#             name="Gget Nucleotide Agent",
#             description=DESCRIPTION,
#             icon="https://raw.githubusercontent.com/acislab/ichatbio/refs/heads/main/icon.svg",
#             documentation_url="https://github.com/acislab/ichatbio-data-handler-agent",
#             url=os.getenv("URL", None),
#             entrypoints=[
#                 AgentEntrypoint(
#                     id="process_request",
#                     description="Processes a user request to retrieve nucleotide data from large nucleotide databases and identify species.",
#                     parameters=EntrypointParameters,
#                 )
#             ],
#         )

#     @override
#     async def run(
#         self,
#         context: ResponseContext,
#         request: str,
#         entrypoint: str,
#         params: EntrypointParameters,
#         metadata: dict[str, Any] | None = None,
#     ):
#         """
#         Running this agent first builds a LangChain agent graph (a loop that alternates between decision-making and
#         tool execution), then executes the graph with `request` as input. The tools themselves are responsible for
#         sending messages back to iChatBio via the `context` object. To give the tools access to the request context, we
#         instantiate new tools each time a request is received; this allows the agent to safely handle concurrent
#         requests.
#         """
#         set_llm_credentials(metadata)

#         artifacts = ArtifactRegistry(params.artifacts)

#         current_request.set(request)
#         current_context.set(context)
#         current_artifacts.set(artifacts)

#         # Build the agent's tools

#         @tool(return_direct=True)  # This tool ends the agent loop
#         async def abort(reason: str):
#             """If you can't fulfill the user's request, abort instead and explain why."""
#             await context.reply(reason)

#         @tool(return_direct=True)  # This tool ends the agent loop
#         async def finish(message: str):
#             """Mark the user's request as successfully completed."""
#             await context.reply(message)

#         tools = [
#             extract_parameters,
#             get_sequence,
#             run_blast_sync,
#             abort,
#             finish,
#         ]

#         # Build a LangChain agent graph

#         llm_credentials = get_llm_credentials()
#         llm = ChatOpenAI(
#             model=os.getenv("LLM"),
#             tool_choice="required",
#             openai_api_key=llm_credentials["api_key"],
#             openai_api_base=llm_credentials["base_url"],
#         )

#         system_message = make_system_message(params.artifacts)
#         agent = langchain.agents.create_agent(
#             model=llm,
#             tools=tools,
#             system_prompt=system_message,
#         )

#         # Run the graph

#         await agent.ainvoke(
#             {
#                 "messages": [
#                     {"role": "user", "content": request},
#                 ]
#             }
#         )


# SYSTEM_MESSAGE = """
# You are a gget nucleotide agent. You have access to the following tools to help you fulfill the user's request:

# {artifacts}

# If you are unable to fulfill the user's request using your available tools, abort and explain why.
# """


# def list_artifact(artifact: Artifact):
#     return f"""\
# - local_id: {artifact.local_id}
#   description: {artifact.description}
#   uris: {artifact.uris}
#   metadata: {artifact.metadata}\
# """


# def make_system_message(artifacts: Iterable[Artifact]):
#     return SYSTEM_MESSAGE.format(
#         artifacts=(
#             "\n\n".join([list_artifact(artifact) for artifact in artifacts])
#             if artifacts
#             else "NO AVAILABLE ARTIFACTS"
#         )
#     ).strip()


# def create_app() -> Starlette:
#     dotenv.load_dotenv()

#     if os.getenv("LLM") is None:
#         raise ValueError("LLM environment variable must be set")

#     agent = GgetNucleotideAgent()
#     app = build_agent_app(agent)
#     return app
