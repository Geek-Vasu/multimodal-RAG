import streamlit as st
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
from backend.app_service import run_fashion_agent
from pathlib import Path
if "backend_result" not in st.session_state:
    st.session_state["backend_result"] = None

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RAG-Drip",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

if "mode" not in st.session_state:
    st.session_state["mode"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# ============================================================================
# THEME VARIABLES & STYLING
# ============================================================================
def get_theme_css():
    theme = st.session_state["theme"]

    if theme == "dark":
        colors = {
            "bg": "#0a0a0a",
            "card_bg": "#1a1a1a",
            "card_hover": "#252525",
            "text": "#e8e8e8",
            "muted": "#9ca3af",
            "accent": "#8b5cf6",
            "accent_hover": "#7c3aed",
            "border": "#2a2a2a",
            "avatar_bg": "#2d1b4e",
            "message_user": "#1e1e1e",
            "message_assistant": "#1a1a2e",
            "gradient_from": "#8b5cf6",
            "gradient_to": "#ec4899"
        }
    else:
        colors = {
            "bg": "#f9fafb",
            "card_bg": "#ffffff",
            "card_hover": "#f3f4f6",
            "text": "#1f2937",
            "muted": "#6b7280",
            "accent": "#8b5cf6",
            "accent_hover": "#7c3aed",
            "border": "#e5e7eb",
            "avatar_bg": "#ede9fe",
            "message_user": "#f3f4f6",
            "message_assistant": "#faf5ff",
            "gradient_from": "#8b5cf6",
            "gradient_to": "#ec4899"
        }

    return f"""
    <style>
        :root {{
            --bg: {colors['bg']};
            --card-bg: {colors['card_bg']};
            --card-hover: {colors['card_hover']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --accent: {colors['accent']};
            --accent-hover: {colors['accent_hover']};
            --border: {colors['border']};
            --avatar-bg: {colors['avatar_bg']};
            --message-user: {colors['message_user']};
            --message-assistant: {colors['message_assistant']};
            --gradient-from: {colors['gradient_from']};
            --gradient-to: {colors['gradient_to']};
        }}

        .stApp {{
            background-color: var(--bg);
            color: var(--text);
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Hide fullscreen button on images */
        button[title="View fullscreen"] {{
            display: none !important;
        }}
        
        /* Header styling */
        .main-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .app-logo {{
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
        }}
        
        .app-title {{
            background: linear-gradient(135deg, var(--gradient-from), var(--gradient-to));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .app-subtitle {{
            color: var(--muted);
            font-size: 1.1rem;
        }}
        
        /* Section headers */
        .section-header {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text);
        }}
        
        /* Divider */
        .divider {{
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
            margin: 32px 0;
            opacity: 0.3;
        }}
        
        /* Info box */
        .info-box {{
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.1));
            border: 1px solid var(--accent);
            border-radius: 12px;
            padding: 16px 20px;
            margin: 16px 0;
        }}
        
        /* Workflow steps */
        .workflow-container {{
            display: flex;
            gap: 12px;
            margin: 16px 0;
        }}
        
        .workflow-step {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            flex: 1;
            text-align: center;
        }}
        
        .workflow-number {{
            background: linear-gradient(135deg, var(--gradient-from), var(--gradient-to));
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        /* Uploaded image */
        .uploaded-image-container {{
            background: var(--card-bg);
            border: 2px solid var(--accent);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            max-width: 650px;
            margin: 0 auto;
            box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2);
        }}
        
        /* Chat container */
        .chat-container {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-top: 16px;
            min-height: 300px;
        }}
        
        /* Style buttons */
        .stButton > button {{
            border-radius: 12px !important;
            transition: all 0.3s ease;
            font-weight: 600;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }}
        
        /* Mode selection buttons - make them stand out */
        div[data-testid="column"] .stButton > button {{
            background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
            color: white !important;
            border: none !important;
            padding: 16px 24px;
            font-size: 1.1rem;
        }}
        
        div[data-testid="column"] .stButton > button:hover {{
            background: linear-gradient(135deg, #7c3aed, #db2777) !important;
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4) !important;
        }}
    </style>
    """

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def toggle_theme():
    st.session_state["theme"] = (
        "light" if st.session_state["theme"] == "dark" else "dark"
    )

def select_mode(mode):
    """Reset contextual state when mode changes"""
    if st.session_state.get("mode") != mode:
        st.session_state["mode"] = mode
        st.session_state["messages"] = []
        st.session_state["uploaded_image"] = None

def get_image_base64(image):
    """Convert PIL image to base64 for display"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# ============================================================================
# APPLY THEME
# ============================================================================
st.markdown(get_theme_css(), unsafe_allow_html=True)

# ============================================================================
# 1. TOP BAR
# ============================================================================
col1, col2 = st.columns([5, 1])

with col1:
    st.markdown('''
    <div class="main-header">
        <div class="app-logo">👟</div>
        <div>
            <div class="app-title">RAG-Drip</div>
            <div class="app-subtitle">Your AI-Powered Fashion Intelligence</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    theme_icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
    if st.button(theme_icon, key="theme_btn"):
        toggle_theme()
        st.rerun()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================================
# 2. HERO / INTENT SECTION - USING NATIVE STREAMLIT
# ============================================================================
st.markdown('<div class="section-header">🎯 What are you looking for today?</div>', unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)

with col1:
    # Mode card using container
    with st.container():
        # Visual card using HTML that WILL render
        is_selected = st.session_state["mode"] == "find_similar"
        
        # Create the icon with gradient background
        st.markdown("""
        <div style='text-align: center; margin-bottom: 16px;'>
            <div style='
                background: linear-gradient(135deg, #8b5cf6, #ec4899);
                width: 80px;
                height: 80px;
                border-radius: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 3rem;
                box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
            '>🔍</div>
        </div>
        <div style='text-align: center; font-size: 1.3rem; font-weight: 600; color: var(--text); margin-bottom: 16px;'>
            Find Similar Shoes
        </div>
        """, unsafe_allow_html=True)
        
        # Selection button
        btn_text = "✓ Selected" if is_selected else "Select This Option"
        if st.button(btn_text, key="find_similar", use_container_width=True):
            select_mode("find_similar")
            st.rerun()

with col2:
    # Mode card using container
    with st.container():
        # Visual card using HTML that WILL render
        is_selected = st.session_state["mode"] == "match_outfit"
        
        # Create the icon with gradient background  
        st.markdown("""
        <div style='text-align: center; margin-bottom: 16px;'>
            <div style='
                background: linear-gradient(135deg, #8b5cf6, #ec4899);
                width: 80px;
                height: 80px;
                border-radius: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 3rem;
                box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
            '>👕</div>
        </div>
        <div style='text-align: center; font-size: 1.3rem; font-weight: 600; color: var(--text); margin-bottom: 16px;'>
            Match Shoes With Outfit
        </div>
        """, unsafe_allow_html=True)
        
        # Selection button
        btn_text = "✓ Selected" if is_selected else "Select This Option"
        if st.button(btn_text, key="match_outfit", use_container_width=True):
            select_mode("match_outfit")
            st.rerun()

st.write("")

# ============================================================================
# 3. WORKFLOW GUIDANCE
# ============================================================================
if st.session_state["mode"]:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    mode_text = "similar shoes" if st.session_state["mode"] == "find_similar" else "matching shoes for your outfit"
    
    st.markdown(f'''
    <div class="info-box">
        <strong>💡 How it works:</strong> You can chat anytime! For best results: Upload an image → Ask specific questions → Get personalized recommendations for {mode_text}
    </div>
    ''', unsafe_allow_html=True)
    
    # Workflow steps
    st.markdown('''
    <div class="workflow-container">
        <div class="workflow-step">
            <div class="workflow-number">1</div>
            <div><strong>Upload Image</strong></div>
            <small style="color: var(--muted);">Share your photo</small>
        </div>
        <div class="workflow-step">
            <div class="workflow-number">2</div>
            <div><strong>Ask Questions</strong></div>
            <small style="color: var(--muted);">Chat with AI stylist</small>
        </div>
        <div class="workflow-step">
            <div class="workflow-number">3</div>
            <div><strong>Get Results</strong></div>
            <small style="color: var(--muted);">Receive recommendations</small>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# 4. IMAGE UPLOAD
# ============================================================================
if st.session_state["mode"]:
    st.write("")
    
    mode_text = (
        "shoes you like"
        if st.session_state["mode"] == "find_similar"
        else "your outfit"
    )

    st.markdown(f'<div class="section-header">📸 Upload Image</div>', unsafe_allow_html=True)
    st.caption(f"Upload an image of {mode_text} to get started")

    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        help=f"Upload a clear photo of {mode_text}"
    )

    if uploaded_file:
     st.session_state["uploaded_image"] = uploaded_file
     image = Image.open(uploaded_file)
     with st.spinner("🧠 Analyzing image with AI stylist..."):
        backend_result = run_fashion_agent(
            mode=st.session_state["mode"],
            image=image,
            text=None
        )   
    
     st.session_state["backend_result"] = backend_result
    
    # Convert to base64 and display without fullscreen
     img_base64 = get_image_base64(image)
        
        # Display image in container
     st.markdown(f'''
        <div class="uploaded-image-container">
            <img src="{img_base64}" style="width: 100%; max-width: 600px; border-radius: 12px; display: block; margin: 0 auto;"/>
            <div style="margin-top: 16px; color: var(--muted); font-size: 0.9rem;">
                ✓ Image uploaded successfully
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Auto-add initial message
     if len(st.session_state["messages"]) == 0:
            if st.session_state["mode"] == "find_similar":
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "👋 Great! I can see your shoe image. I can help you find similar styles, different colors, or recommendations based on the style you uploaded. What would you like to know?"
                })
            else:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "👋 Perfect! I can see your outfit. I can recommend shoes that would match perfectly, suggest styles for different occasions, or help you choose the right color. What are you looking for?"
                })
# ============================================================================
# 6. BACKEND RESULTS (NEW)
# ============================================================================
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("images")  # must match your indexing folder
if st.session_state.get("backend_result"):
    result = st.session_state["backend_result"]
    llm_output = result.get("llm_output", {})

    summary = llm_output.get("summary", "No explanation available.")
    recommended = llm_output.get("recommended", [])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 🧠 AI Recommendation")

    # ✅ Grounded LLM explanation
    st.markdown(
        f'<div class="info-box">{summary}</div>',
        unsafe_allow_html=True
    )

    # ✅ Grounded recommendations (TEXT ONLY)
    if recommended:
        st.markdown("### 👟 Recommended Shoes")

        for item in recommended[:6]:
            st.markdown(
                f"""
                <div class="product-card">
                    <b>{item.get("brand", "Sneaker").title()}</b><br>
                    File: <code>{item.get("filename")}</code><br>
                    Match confidence: <b>{round(item.get("confidence", 0), 2)}</b>
                </div>
                """,
                unsafe_allow_html=True
            )


        
# ============================================================================
# 5. AI STYLIST CHAT
# ============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">💬 AI Stylist Chat</div>', unsafe_allow_html=True)
st.caption("Ask questions grounded in the uploaded image and AI recommendations")

# Display chat messages
for msg in st.session_state["messages"]:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the recommendation..."):
    # Store user message
    st.session_state["messages"].append(
        {"role": "user", "content": prompt}
    )

    # Backend-grounded response ONLY
    if st.session_state.get("backend_result"):
        llm_output = st.session_state["backend_result"].get("llm_output", {})
        assistant_response = llm_output.get(
            "summary",
            "I’ve analyzed the image, but no explanation is available."
        )
    else:
        assistant_response = (
            "Please upload an image first so I can analyze it and give grounded recommendations."
        )

    # Store assistant response
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_response}
    )

    st.rerun()