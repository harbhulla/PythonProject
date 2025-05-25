import os
import getpass
import re
from dotenv import load_dotenv
import openai

import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize OpenAI client for moderation
openai.api_key = os.getenv("OPENAI_API_KEY")


# OpenAI Moderation API integration
class OpenAIModerationFilter:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def moderate_content(self, text):
        """Use OpenAI's Moderation API to check content"""
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]

            if result.flagged:
                # Get the specific categories that were flagged
                flagged_categories = []
                categories = result.categories

                if categories.hate: flagged_categories.append("hate")
                if categories.hate_threatening: flagged_categories.append("hate/threatening")
                if categories.harassment: flagged_categories.append("harassment")
                if categories.harassment_threatening: flagged_categories.append("harassment/threatening")
                if categories.self_harm: flagged_categories.append("self-harm")
                if categories.self_harm_intent: flagged_categories.append("self-harm/intent")
                if categories.self_harm_instructions: flagged_categories.append("self-harm/instructions")
                if categories.sexual: flagged_categories.append("sexual")
                if categories.sexual_minors: flagged_categories.append("sexual/minors")
                if categories.violence: flagged_categories.append("violence")
                if categories.violence_graphic: flagged_categories.append("violence/graphic")

                return True, f"Content flagged for: {', '.join(flagged_categories)}"

            return False, None

        except Exception as e:
            st.warning(f"Moderation API error: {str(e)}")
            # Fallback to basic filtering if API fails
            return False, None

    def moderate_with_context(self, text, allow_medical_context=True):
        """Moderate content with consideration for medical/therapeutic context"""
        is_flagged, reason = self.moderate_content(text)

        if is_flagged and allow_medical_context:
            # Check if this might be legitimate medical/therapeutic content
            medical_indicators = [
                'therapy', 'treatment', 'counseling', 'psychiatric', 'clinical',
                'mental health', 'depression', 'anxiety', 'cbt', 'cognitive behavioral',
                'therapeutic', 'psychology', 'psychotherapy', 'medication', 'diagnosis'
            ]

            text_lower = text.lower()
            medical_context_score = sum(1 for term in medical_indicators if term in text_lower)

            # If strong medical context, allow with warning
            if medical_context_score >= 2:
                st.info(f"Content flagged but allowed due to medical context: {reason}")
                return False, f"Allowed with medical context warning: {reason}"

        return is_flagged, reason


# Enhanced content filtering system
class ContentFilter:
        def __init__(self):
            # Initialize OpenAI moderation
            self.openai_moderator = OpenAIModerationFilter()

            # Harmful/inappropriate keywords to filter out (backup to OpenAI moderation)
            self.blocked_keywords = [
                # Violence and self-harm
                'suicide', 'kill yourself', 'self-harm', 'cutting', 'overdose',
                'violence', 'murder', 'assault', 'abuse', 'torture',
                # Drugs and substances (except medical context)
                'illegal drugs', 'drug dealing', 'cocaine', 'heroin', 'meth',
                # Inappropriate sexual content
                'explicit sexual', 'pornography', 'sexual abuse',
                # Hate speech indicators
                'racial slur', 'hate speech', 'discrimination',
                # Conspiracy theories and misinformation
                'conspiracy theory', 'fake news', 'misinformation'
            ]

            # Medical/therapeutic terms that should be allowed
            self.allowed_medical_terms = [
                'medication', 'prescription', 'therapy', 'treatment', 'counseling',
                'depression', 'anxiety', 'mental health', 'psychiatric', 'clinical',
                'cognitive behavioral', 'cbt', 'therapeutic', 'psychology'
            ]

        def comprehensive_content_check(self, text, context="general"):
            """Comprehensive content filtering using OpenAI + custom rules"""

            # First, use OpenAI's moderation API
            is_flagged, openai_reason = self.openai_moderator.moderate_with_context(
                text,
                allow_medical_context=(context == "medical")
            )

            if is_flagged:
                return True, f"OpenAI Moderation: {openai_reason}"

            # Fallback to custom keyword filtering
            is_harmful, custom_reason = self.contains_harmful_content(text)
            if is_harmful:
                return True, f"Custom Filter: {custom_reason}"

            return False, None

        def contains_harmful_content(self, text):
            """Custom keyword-based filtering (fallback)"""
            text_lower = text.lower()

            # Check for blocked keywords
            for keyword in self.blocked_keywords:
                if keyword in text_lower:
                    # Check if it's in a medical/therapeutic context
                    context_window = 100  # characters around the keyword
                    keyword_pos = text_lower.find(keyword)
                    context_start = max(0, keyword_pos - context_window)
                    context_end = min(len(text_lower), keyword_pos + len(keyword) + context_window)
                    context = text_lower[context_start:context_end]

                    # Allow if surrounded by medical terms
                    medical_context = any(term in context for term in self.allowed_medical_terms)
                    if not medical_context:
                        return True, f"Potentially harmful content detected: {keyword}"

            return False, None

        def filter_text_chunks(self, chunks):
            """Filter chunks using comprehensive moderation"""
            filtered_chunks = []
            filtered_reasons = []

            progress_bar = st.progress(0)
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                # Update progress
                progress_bar.progress((i + 1) / total_chunks)

                # Apply comprehensive filtering
                is_harmful, reason = self.comprehensive_content_check(chunk, context="medical")

                if not is_harmful:
                    filtered_chunks.append(chunk)
                else:
                    filtered_reasons.append(reason)
                    with st.expander(f"Filtered Chunk {i + 1}"):
                        st.warning(f"Reason: {reason}")
                        st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)

            progress_bar.empty()

            st.info(f"Content filtering: {len(filtered_reasons)} chunks filtered out of {total_chunks} total")
            if filtered_reasons:
                with st.expander("Filtering Summary"):
                    for reason in set(filtered_reasons):
                        count = filtered_reasons.count(reason)
                        st.write(f"• {reason}: {count} chunks")

            return filtered_chunks

        def is_query_appropriate(self, query):
            """Check if user query is appropriate using comprehensive filtering"""

            # Use comprehensive content check
            is_inappropriate, reason = self.comprehensive_content_check(query, context="general")
            if is_inappropriate:
                return False, reason

            # Check for off-topic requests (custom rules)
            off_topic_patterns = [
                r'how to (make|create|build).*(bomb|weapon|drug)',
                r'illegal (activities|ways|methods)',
                r'hack(ing)? (into|someone)',
                r'personal information about',
                r'generate (fake|false) (documents|id|credentials)'
            ]

            for pattern in off_topic_patterns:
                if re.search(pattern, query.lower()):
                    return False, "Query appears to be requesting inappropriate or illegal information"

            return True, None


# Initialize content filter
content_filter = ContentFilter()

# Set Pinecone API key
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)
index_name = "langchainv2"


# Cache index creation
@st.cache_resource
def get_pinecone_index():
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


# Cache vector store + embeddings
@st.cache_resource
def get_vector_store():
    index = get_pinecone_index()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return PineconeVectorStore(index=index, embedding=embeddings)


# Cache document loading and processing with content filtering
@st.cache_data
def process_pdfs():
    pdf_files = [
        'cbt guide en.pdf',
        'therapists_guide_to_brief_cbtmanual.pdf',
        '2cacbt-training-guidelines-final-eng.pdf'
    ]

    # Initialize vector store
    vector_store = get_vector_store()

    # Check if we need to add documents (by checking if index exists and has documents)
    try:
        test_results = vector_store.as_retriever().invoke("cognitive behavioral therapy")
        if test_results:
            # Documents are already in the index
            st.success("Using existing document embeddings from Pinecone.")
            return True
    except Exception:
        # If error occurs, we'll proceed to add documents
        pass

    # Process and add documents with content filtering
    all_chunks = []
    total_filtered = 0

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                # Basic text cleaning
                page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                page_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', page_text)  # Remove special chars
                text += page_text

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200
            )
            chunks = splitter.split_text(text)

            # Apply content filtering
            filtered_chunks = content_filter.filter_text_chunks(chunks)
            all_chunks.extend(filtered_chunks)

            filtered_count = len(chunks) - len(filtered_chunks)
            total_filtered += filtered_count

            st.success(f"Processed {pdf}: {len(filtered_chunks)} clean chunks extracted ({filtered_count} filtered)")

        except Exception as e:
            st.error(f"Error processing {pdf}: {str(e)}")

    # Add to vector store if we have chunks
    if all_chunks:
        uuids = [f"id{i}" for i in range(1, len(all_chunks) + 1)]
        try:
            vector_store.add_texts(texts=all_chunks, ids=uuids)
            st.success(f"Added {len(all_chunks)} filtered text chunks to Pinecone (Total filtered: {total_filtered})")
            return True
        except Exception as e:
            st.error(f"Error adding documents to Pinecone: {str(e)}")
            return False
    else:
        st.error("No clean text chunks were extracted from PDFs after filtering")
        return False


# Enhanced response filtering with OpenAI moderation
def filter_ai_response(response):
    """Enhanced filtering for AI responses using OpenAI moderation"""

    # Use OpenAI moderation on the response
    moderator = OpenAIModerationFilter()
    is_flagged, reason = moderator.moderate_content(response)

    if is_flagged:
        st.warning(f"Response was moderated: {reason}")
        return "I apologize, but I cannot provide that information. Please ask about cognitive behavioral therapy or mental health topics in a constructive way."

    # Check if response contains harmful content (backup check)
    is_harmful, custom_reason = content_filter.contains_harmful_content(response)
    if is_harmful:
        st.warning(f"Response filtered by custom rules: {custom_reason}")
        return "I apologize, but I cannot provide that information. Please ask about cognitive behavioral therapy or mental health topics in a constructive way."

    # Ensure response stays on topic
    if not any(term in response.lower() for term in
               ['therapy', 'cbt', 'mental health', 'cognitive', 'behavioral', 'treatment', 'counseling',
                'psychological']):
        return "I'm designed to help with questions about cognitive behavioral therapy and mental health. Could you please rephrase your question to focus on these topics?"

    return response


# Setup chat interface
st.title("🧠 Mental Health Chatbot")
st.markdown("*A safe space for CBT and mental health information*")

# Content filtering info
with st.sidebar:
    st.header("🛡️ Content Safety")
    st.info(
        "This chatbot uses OpenAI's Moderation API plus custom filtering to ensure safe, therapeutic conversations.")

    with st.expander("Safety Features"):
        st.markdown("""
        - ✅ **OpenAI Moderation API** - Professional content screening
        - ✅ **Medical Context Awareness** - Allows therapeutic discussions
        - ✅ **Custom Keyword Filtering** - Additional safety layer
        - ✅ **Response Monitoring** - All AI responses are checked
        - ✅ **Crisis Detection** - Identifies when professional help is needed
        - ✅ **Real-time Filtering** - Content checked before processing
        """)

    # Moderation API status
    try:
        test_moderator = OpenAIModerationFilter()
        test_result = test_moderator.moderate_content("This is a test.")
        st.success("✅ OpenAI Moderation API: Active")
    except Exception as e:
        st.error(f"❌ OpenAI Moderation API: Error - {str(e)}")
        st.warning("Using fallback filtering only")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_message" not in st.session_state:
    st.session_state.system_message = SystemMessage(
        "You are a helpful and safe assistant for cognitive behavioral therapy and mental health questions. Always prioritize user safety and well-being.")

# Display chat history (exclude system message from UI)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Process PDFs and add to vector store
with st.sidebar:
    st.header("📄 Document Processing")
    docs_loaded = process_pdfs()
    if docs_loaded:
        st.success("Documents are loaded and ready!")
    else:
        st.error("Document processing failed. Check file paths and API keys.")

# Chat input with content filtering
prompt = st.chat_input("Ask about cognitive behavioral therapy or mental health...")

if prompt:
    # First, use comprehensive moderation on the user query
    with st.spinner("Checking content safety..."):
        is_appropriate, filter_reason = content_filter.is_query_appropriate(prompt)

    if not is_appropriate:
        # Display filtered query warning
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

        with st.chat_message("assistant"):
            if "self-harm" in filter_reason.lower() or "suicide" in filter_reason.lower():
                crisis_msg = """I'm concerned about your wellbeing. If you're having thoughts of self-harm or suicide, please reach out for immediate help:

**Crisis Resources:**
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741  
- **Emergency Services**: 911

I'm here to provide information about cognitive behavioral therapy and mental health resources, but professional support is important for crisis situations."""
                st.markdown(crisis_msg)
                st.session_state.messages.append(AIMessage(crisis_msg))
            else:
                warning_msg = f"I cannot assist with that request. {filter_reason}. Please ask about cognitive behavioral therapy, mental health, or related therapeutic topics."
                st.markdown(warning_msg)
                st.session_state.messages.append(AIMessage(warning_msg))

    else:
        # Add user message to UI and history
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

        if docs_loaded:
            # Retrieve relevant documents
            with st.spinner("Searching documents..."):
                retriever = get_vector_store().as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 3, "score_threshold": 0.5},
                )
                docs = retriever.invoke(prompt)

            if docs:
                docs_text = "\n\n".join([d.page_content for d in docs])

                # Display retrieved context in sidebar
                with st.sidebar:
                    with st.expander("Retrieved Context"):
                        st.markdown(docs_text)

                # Generate response using LLM with enhanced safety prompt
                system_prompt = f"""You are a helpful and responsible assistant for cognitive behavioral therapy and mental health questions. 

IMPORTANT SAFETY GUIDELINES:
- Only provide information based on the provided documents
- Focus exclusively on therapeutic, educational, and supportive content
- Never provide medical diagnoses or replace professional treatment
- If asked about self-harm or crisis situations, direct users to professional help
- Stay within the scope of CBT and mental health education
- Be empathetic but maintain professional boundaries

Use ONLY the context below to answer the user's question. 
If the answer is not in the documents, say "I don't have that information in my knowledge base, but I'd recommend consulting with a mental health professional."

Context: {docs_text}
"""
                try:
                    with st.spinner("Thinking..."):
                        # Create proper message sequence for LLM
                        conversation_messages = [
                                                    SystemMessage(system_prompt)
                                                ] + st.session_state.messages[
                                                    -5:]  # Include last 5 messages for context

                        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                        result = llm.invoke(conversation_messages).content

                    # Apply response filtering
                    filtered_result = filter_ai_response(result)

                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(filtered_result)
                    st.session_state.messages.append(AIMessage(filtered_result))
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                with st.chat_message("assistant"):
                    no_docs_msg = "I couldn't find relevant information about that in my knowledge base. For mental health support, I recommend consulting with a qualified mental health professional or calling a crisis helpline if you're in immediate need."
                    st.markdown(no_docs_msg)
                st.session_state.messages.append(AIMessage(no_docs_msg))
        else:
            with st.chat_message("assistant"):
                error_msg = "I'm having trouble accessing my knowledge base. Please check the document processing status in the sidebar."
                st.markdown(error_msg)
            st.session_state.messages.append(AIMessage(error_msg))

# Crisis resources in sidebar
with st.sidebar:
    st.header("🆘 Crisis Resources")
    st.error("**If you're in crisis, please reach out:**")
    st.markdown("""
    - **National Suicide Prevention Lifeline**: 988
    - **Crisis Text Line**: Text HOME to 741741  
    - **Emergency Services**: 911
    - **SAMHSA Helpline**: 1-800-662-4357
    """)