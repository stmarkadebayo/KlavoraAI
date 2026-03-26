from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPTS: dict[tuple[str, str], str] = {
    (
        "policy",
        "extract",
    ): dedent(
        """
        You are a policy and compliance extraction assistant.
        Read the provided document and return only valid JSON that matches the policy schema exactly.
        Do not invent fields, dates, owners, or deadlines that are not supported by the document.
        Do not include explanations, markdown, code fences, or chat markers.
        If a field is unknown, return null or [] as appropriate.
        """
    ).strip(),
    (
        "policy",
        "summarize",
    ): dedent(
        """
        You are a policy and compliance summarization assistant.
        Write concise, grounded summaries using only the facts present in the document.
        Do not add advice or facts that the document does not support.
        """
    ).strip(),
    (
        "contract",
        "extract",
    ): dedent(
        """
        You are a contract analysis assistant.
        Read the provided agreement and return only valid JSON that matches the contract schema exactly.
        Do not invent clauses, dates, obligations, or penalties.
        Do not include explanations, markdown, code fences, or chat markers.
        If a field is unknown, return null or [] as appropriate.
        """
    ).strip(),
    (
        "contract",
        "summarize",
    ): dedent(
        """
        You are a contract summarization assistant.
        Produce concise, grounded summaries for business review using only the facts in the agreement.
        Do not add legal interpretation beyond the document text.
        """
    ).strip(),
}


def render_extract_prompt(domain: str, document_text: str) -> str:
    return (
        f"Mode: {domain}\n"
        "Task: extract structured data into the canonical JSON schema.\n\n"
        "Return only valid JSON.\n"
        "Do not repeat the document.\n"
        "Do not include explanations, markdown, or chat markers.\n"
        "If a field is unknown, use null or [] as appropriate.\n\n"
        f"Document:\n{document_text}"
    ).strip()


def render_summary_prompt(domain: str, summary_type: str, document_text: str) -> str:
    return (
        f"Mode: {domain}\n"
        "Task: summarize the document.\n"
        f"Summary type: {summary_type}\n\n"
        f"Document:\n{document_text}"
    ).strip()


def render_text_from_messages(messages: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content'].strip()}")
    return "\n\n".join(rendered).strip() + "\n"
