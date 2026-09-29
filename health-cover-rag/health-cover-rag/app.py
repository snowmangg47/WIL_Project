import streamlit as st

from rag import (
    retrieve,
    generate_answer
)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Health Cover RAG Assistant",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🏥 Health Cover RAG Assistant"
)

st.write(
    """
    Ask questions about the health cover information
    contained in the supplied insurance documents.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "RAG Settings"
)


# Retrieval method
retrieval_option = st.sidebar.selectbox(
    "Retrieval Method",
    [
        "Dense FAISS",
        "BM25"
    ]
)


if retrieval_option == "Dense FAISS":

    retrieval_method = "dense"

else:

    retrieval_method = "bm25"


# Top-k
top_k = st.sidebar.slider(
    "Number of retrieved chunks (Top-k)",
    min_value=1,
    max_value=5,
    value=5
)


st.sidebar.markdown(
    "---"
)


st.sidebar.subheader(
    "Current Configuration"
)


st.sidebar.write(
    f"**Retriever:** {retrieval_option}"
)

st.sidebar.write(
    f"**Top-k:** {top_k}"
)

st.sidebar.write(
    "**Embedding Model:** all-MiniLM-L6-v2"
)

st.sidebar.write(
    "**LLM:** Llama 3.2 3B"
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER QUESTION
# =========================================================

question = st.chat_input(
    "Ask a health cover question..."
)


if question:


    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # Retrieve evidence
    # -----------------------------------------------------

    with st.spinner(
        "Searching health cover documents..."
    ):

        retrieved_chunks = retrieve(
            question,
            top_k=top_k,
            method=retrieval_method
        )


        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = generate_answer(
            question,
            retrieved_chunks
        )


    # -----------------------------------------------------
    # Display answer
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )


        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        if retrieved_chunks:

            st.markdown(
                "### 📚 Sources"
            )


            shown_sources = set()


            for result in retrieved_chunks:

                source_key = (
                    result["source"],
                    result["page"]
                )


                if source_key not in shown_sources:

                    st.write(
                        f"• **{result['source']}** "
                        f"— Page {result['page']}"
                    )


                    shown_sources.add(
                        source_key
                    )


        # -------------------------------------------------
        # Retrieved evidence
        # -------------------------------------------------

        if retrieved_chunks:

            with st.expander(
                "🔎 View Retrieved Evidence"
            ):

                for number, result in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"### Result {number}"
                    )

                    st.write(
                        f"**Document:** "
                        f"{result['source']}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{result['page']}"
                    )

                    st.write(
                        f"**Retrieval Score:** "
                        f"{result['score']:.4f}"
                    )

                    st.write(
                        "**Retrieved Text:**"
                    )

                    st.write(
                        result["text"]
                    )

                    st.markdown(
                        "---"
                    )


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# =========================================================
# CLEAR CHAT
# =========================================================

st.sidebar.markdown(
    "---"
)


if st.sidebar.button(
    "Clear Chat"
):

    st.session_state.messages = []

    st.rerun()