import streamlit as st
import pandas as pd
from google import genai
import os
from dotenv import load_dotenv
import chromadb
import uuid
from datetime import datetime, timedelta
import urllib.parse

# Load environment variables
load_dotenv()

# Enhanced CSS for better UX
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Card styling for metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Report styling */
    .report-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .report-header {
        color: #1f77b4;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .priority-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border-left: 3px solid #ff6b6b;
    }
    
    .coach-advice {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .message-preview {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #cce7ff;
        margin: 1rem 0;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class SalesCRM:
    def __init__(self):
        self.setup_gemini()
        self.setup_vector_db()
        self.data_loaded = False
        
    def setup_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in .env file")
            st.stop()
        self.client = genai.Client(api_key=api_key)
    
    def setup_vector_db(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="sales_leads",
            metadata={"description": "Sales leads and activities"}
        )
    
    def load_all_sheets(self, uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                sheets = {'Daily Lead Log': pd.read_csv(uploaded_file)}
            else:
                sheets = pd.read_excel(uploaded_file, sheet_name=None)
            
            if not self.data_loaded:
                daily_log = sheets['Daily Lead Log']
                self.store_leads_in_db(daily_log)
                self.data_loaded = True
            
            return sheets
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def get_embeddings(self, text):
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return None
    
    def store_leads_in_db(self, df):
        documents = []
        embeddings = []
        ids = []
        
        with st.status("Storing leads in vector database...", expanded=True) as status:
            for idx, row in df.iterrows():
                try:
                    doc_text = f"""
                    Lead: {row.get('Name', 'N/A')}
                    Company: {row.get('Company', 'N/A')}
                    Title: {row.get('Title', 'N/A')}
                    Source: {row.get('Source', 'N/A')}
                    Action Taken: {row.get('Action Taken', 'N/A')}
                    Next Step: {row.get('Next Step', 'N/A')}
                    Status: {row.get('Status Stage', 'N/A')}
                    Sales Rep: {row.get('Sales Rep', 'N/A')}
                    Notes: {row.get('Notes', '')}
                    Due Date: {row.get('Due Date', '')}
                    """
                    
                    embedding = self.get_embeddings(doc_text)
                    if embedding:
                        documents.append(doc_text)
                        embeddings.append(embedding)
                        ids.append(str(uuid.uuid4()))
                    
                except Exception as e:
                    continue
            
            if documents:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids
                )
                status.update(label=f"✅ Stored {len(documents)} leads in vector database", state="complete")
    
    def analyze_with_ai(self, df, sales_rep=None):
        try:
            rep_filter = f" for {sales_rep}" if sales_rep else ""
            
            prompt = f"""
            Analyze this sales data{rep_filter} and provide SPECIFIC, ACTIONABLE advice in this EXACT structured format:

            ## 🎯 TODAY'S TOP 3 PRIORITIES

            ### 1. [Lead Name - Company]
            **Action Required:** [Specific action needed]
            **Urgency:** [Why this is urgent - specific reason]
            **Expected Outcome:** [What success looks like]

            ### 2. [Lead Name - Company]  
            **Action Required:** [Specific action needed]
            **Urgency:** [Why this is urgent - specific reason]
            **Expected Outcome:** [What success looks like]

            ### 3. [Lead Name - Company]
            **Action Required:** [Specific action needed]
            **Urgency:** [Why this is urgent - specific reason]
            **Expected Outcome:** [What success looks like]

            ## 💡 COACHING RECOMMENDATIONS

            ### Area 1: [Specific improvement area]
            **What to do differently:** [Exactly what to change]
            **Expected impact:** [How this will improve results]

            ### Area 2: [Specific improvement area]
            **What to do differently:** [Exactly what to change]
            **Expected impact:** [How this will improve results]

            ## 📈 QUICK WINS

            ### Quick Win 1: [Lead Name]
            **Next Step:** [Easy action that could close quickly]
            **Timeline:** [When to act]
            **Confidence:** [High/Medium]

            ### Quick Win 2: [Lead Name]
            **Next Step:** [Easy action that could close quickly]
            **Timeline:** [When to act]
            **Confidence:** [High/Medium]

            DATA TO ANALYZE:
            {df.to_string() if len(df) < 15 else df.head(15).to_string()}

            Focus on SPECIFIC names, companies, and ACTIONS from the data provided.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def generate_followup_message(self, lead_data, message_type="default"):
        try:
            if message_type == "connection":
                prompt = f"""
                Create a FIRST CONNECTION LinkedIn message with this structure:

                **Subject:** Connection Request - Mutual Interest in [Industry/Area]

                **Message:**
                Hi [Name],

                I came across your profile and noticed your work at [Company] in [Title]. [Specific compliment about their role/company].

                I'd love to connect and learn more about [specific aspect of their work].

                Best regards,
                [Sales Rep Name]

                **Key points to include:**
                - Professional but friendly tone
                - Specific reference to their company/role
                - Clear but soft call-to-action
                - 2-3 sentences maximum

                Lead Details:
                Name: {lead_data['Name']}
                Company: {lead_data['Company']} 
                Title: {lead_data['Title']}
                Source: {lead_data.get('Source', '')}
                """
            elif message_type == "follow_up_1":
                prompt = f"""
                Create a FIRST FOLLOW-UP message (2-3 days after connection):

                **Subject:** Following up on our connection

                **Message:**
                Hi [Name],

                Hope you're having a productive week. I wanted to follow up on our connection and [provide specific value - share relevant insight/article/case study].

                [Specific question to engage them].

                Looking forward to your thoughts.

                Best,
                [Sales Rep Name]

                Lead Details:
                Name: {lead_data['Name']}
                Company: {lead_data['Company']}
                Last Action: {lead_data.get('Action Taken', 'Connected')}
                Notes: {lead_data.get('Notes', '')}
                """
            elif message_type == "proposal_followup":
                prompt = f"""
                Create a PROFESSIONAL PROPOSAL FOLLOW-UP:

                **Subject:** Following up on our proposal

                **Message:**
                Hi [Name],

                I wanted to follow up on the proposal we sent [timeframe]. Do you have any questions I can clarify?

                [Offer specific additional value - case study, reference, demo]

                Would you be available for a quick call [suggest specific days/times]?

                Best regards,
                [Sales Rep Name]

                Lead Details:
                Name: {lead_data['Name']}
                Company: {lead_data['Company']}
                Days since proposal: {lead_data.get('Days Since Action', 'several')}
                """
            else:
                prompt = f"""
                Create a personalized follow-up message:

                **Structure:**
                - Professional greeting
                - Reference previous interaction
                - Provide specific value
                - Clear call-to-action
                - Professional closing

                Lead Details:
                Name: {lead_data['Name']}
                Company: {lead_data['Company']}
                Title: {lead_data['Title']}
                Last Action: {lead_data.get('Action Taken', '')}
                Next Step: {lead_data.get('Next Step', '')}
                Notes: {lead_data.get('Notes', '')}
                """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Message generation error: {str(e)}"
    
    def generate_manager_report(self, df, sales_rep, start_date, end_date):
        try:
            # Ensure Action Date is datetime and handle NaT
            df['Action Date'] = pd.to_datetime(df['Action Date'], errors='coerce')
            
            # Filter data for the specific rep and date range
            mask = (
                (df['Sales Rep'] == sales_rep) & 
                (df['Action Date'].notna()) &
                (df['Action Date'] >= pd.to_datetime(start_date)) & 
                (df['Action Date'] <= pd.to_datetime(end_date))
            )
            date_data = df[mask]
            
            if date_data.empty:
                return f"No activity found for {sales_rep} from {start_date} to {end_date}"
            
            # Calculate metrics correctly
            total_activities = len(date_data)
            
            # Count specific activities more accurately
            new_leads = len(date_data[
                (date_data['Action Taken'].str.contains('connection|connect|initial', case=False, na=False)) |
                (date_data['Status Stage'] == 'New')
            ])
            
            calls_made = len(date_data[
                date_data['Action Taken'].str.contains('call|phone|discussion', case=False, na=False)
            ])
            
            proposals_sent = len(date_data[
                date_data['Action Taken'].str.contains('proposal|quote|estimate', case=False, na=False) |
                (date_data['Status Stage'] == 'Proposal Sent')
            ])
            
            deals_closed = len(date_data[
                (date_data['Status Stage'] == 'Closed Won') |
                (date_data['Action Taken'].str.contains('closed|won|signed', case=False, na=False))
            ])
            
            # Active pipeline (current state, not just date-filtered)
            current_active = df[
                (df['Sales Rep'] == sales_rep) & 
                (df['Status Stage'].isin(['New', 'Contacted', 'Engaged', 'Proposal Sent', 'Negotiation']))
            ]
            
            # Better conversion rate calculations
            contacted_leads = len(date_data[
                date_data['Status Stage'].isin(['Contacted', 'Engaged', 'Proposal Sent', 'Negotiation'])
            ])
            
            # Safe conversion rate calculations with better logic
            if new_leads > 0:
                lead_to_contact = min(contacted_leads / new_leads, 1.0)
            else:
                lead_to_contact = 0
            
            if contacted_leads > 0:
                contact_to_proposal = min(proposals_sent / contacted_leads, 1.0)
            else:
                contact_to_proposal = 0
            
            # Overall conversion based on actual closed deals vs qualified activities
            qualified_activities = len(date_data[date_data['Status Stage'].isin(['Contacted', 'Engaged', 'Proposal Sent'])])
            if qualified_activities > 0:
                overall_conversion = min((deals_closed / qualified_activities) * 100, 100.0)
            else:
                overall_conversion = 0
            
            prompt = f"""
            Create a CONCISE WhatsApp-friendly manager report with this EXACT structure:

            📊 SALES PERFORMANCE REPORT
            Rep: {sales_rep}
            Period: {start_date} to {end_date}

            🎯 KEY METRICS:
            • Total Activities: {total_activities}
            • New Leads Added: {new_leads}
            • Calls/Discussions: {calls_made}
            • Proposals Sent: {proposals_sent}
            • Deals Closed: {deals_closed}
            • Current Active Pipeline: {len(current_active)}

            📈 CONVERSION INSIGHTS:
            • Lead-to-Contact: {lead_to_contact:.1%}
            • Contact-to-Proposal: {contact_to_proposal:.1%}
            • Win Rate: {overall_conversion:.1f}%

            🚀 TOP ACHIEVEMENTS:
            • [List 1-2 key achievements based on actual metrics above]

            💡 AREAS FOR IMPROVEMENT:
            • [List 1-2 specific, actionable improvements based on metrics]

            🎯 NEXT WEEK FOCUS:
            • [Priority 1 - specific, measurable action]
            • [Priority 2 - specific, measurable action]

            Keep it brief, actionable, and WhatsApp-friendly (max 15 lines total).
            Use only the metrics provided above.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Report generation error: {str(e)}"
    
    def sales_coach_chat(self, query, df, sales_rep):
        try:
            prompt = f"""
            As an expert sales coach with 15+ years experience, provide SPECIFIC, ACTIONABLE advice to {sales_rep}.

            CURRENT PIPELINE SNAPSHOT:
            {df[df['Sales Rep'] == sales_rep][['Name', 'Company', 'Status Stage', 'Action Taken', 'Next Step']].head(8).to_string() if not df.empty else "No current data"}

            QUESTION: {query}

            Structure your response as:

            ## 🎯 IMMEDIATE ACTION PLAN

            ### Step 1: [Specific action]
            **Why:** [Reasoning]
            **How:** [Detailed instructions]

            ### Step 2: [Specific action] 
            **Why:** [Reasoning]
            **How:** [Detailed instructions]

            ## 💡 PRO TIPS
            • [Tip 1 - specific to their situation]
            • [Tip 2 - common pitfall to avoid]
            • [Tip 3 - best practice]

            ## 📝 TEMPLATE (if applicable)
            [Provide ready-to-use template if relevant]

            ## 🎉 MOTIVATION
            [Brief motivational closing]

            Keep it practical and specific to their pipeline data.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Coach error: {str(e)}"
    
    def get_rep_performance(self, df, sales_rep):
        rep_data = df[df['Sales Rep'] == sales_rep]
        
        if rep_data.empty:
            return {}
            
        total_leads = len(rep_data)
        active_leads = len(rep_data[rep_data['Status Stage'].isin(['New', 'Contacted', 'Engaged', 'Proposal Sent'])])
        closed_won = len(rep_data[rep_data['Status Stage'] == 'Closed Won'])
        conversion_rate = (closed_won / total_leads * 100) if total_leads > 0 else 0
        
        return {
            'total_leads': total_leads,
            'active_leads': active_leads,
            'conversion_rate': round(conversion_rate, 1),
            'calls_made': len(rep_data[rep_data['Action Taken'].str.contains('call', case=False, na=False)]),
            'proposals_sent': len(rep_data[rep_data['Action Taken'] == 'Sent proposal']),
            'meetings_booked': len(rep_data[rep_data['Action Taken'] == 'Call scheduled'])
        }

def send_whatsapp_message(phone_number, message):
    """Generate WhatsApp URL with pre-filled message"""
    clean_phone = ''.join(filter(str.isdigit, phone_number))
    encoded_message = urllib.parse.quote(message)
    whatsapp_url = f"https://wa.me/{clean_phone}?text={encoded_message}"
    return whatsapp_url

def main():
    st.set_page_config(
        page_title="Sales CRM AI Assistant", 
        layout="wide",
        page_icon="📊"
    )
    
    # Header
    st.title("Sales CRM AI Assistant")
    st.markdown("Transform your sales process with AI-powered insights and automation")
    
    # Initialize CRM
    if 'crm' not in st.session_state:
        st.session_state.crm = SalesCRM()
    
    crm = st.session_state.crm
    
    # Manager's WhatsApp number (hardcoded)
    MANAGER_WHATSAPP = "+919946294194"  # Change to actual number
    
    # Sidebar
    with st.sidebar:
        st.header("Data Setup")
        uploaded_file = st.file_uploader(
            "Upload Lead Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your CSV or Excel file with sales leads"
        )
        
        if uploaded_file:
            st.success("Data uploaded successfully")
            
        st.markdown("---")
        st.header("Features")
        st.markdown("""
        - **AI Sales Coach** - Personalized guidance
        - **Smart Messaging** - Context-aware messages  
        - **Performance Reports** - Manager-ready insights
        - **Pipeline Analytics** - Data-driven decisions
        """)
    
    # Main content
    if uploaded_file:
        sheets = crm.load_all_sheets(uploaded_file)
        if sheets:
            daily_log = sheets['Daily Lead Log']
            
            # Sales Rep Selection
            sales_reps = daily_log['Sales Rep'].unique()
            selected_rep = st.selectbox("Select Your Profile", sales_reps)
            
            # Performance Overview
            st.subheader("Performance Dashboard")
            perf = crm.get_rep_performance(daily_log, selected_rep)
            
            if perf:
                cols = st.columns(4)
                metrics = [
                    ("Total Leads", perf['total_leads'], "#1f77b4"),
                    ("Active Pipeline", perf['active_leads'], "#2ca02c"), 
                    ("Conversion Rate", f"{perf['conversion_rate']}%", "#ff7f0e"),
                    ("Proposals Sent", perf['proposals_sent'], "#d62728")
                ]
                
                for col, (label, value, color) in zip(cols, metrics):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="color: {color}">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Main Tabs
            tab1, tab2, tab3 = st.tabs(["AI Sales Coach", "Message Generator", "Manager Report"])
            
            with tab1:
                st.subheader("AI Sales Coach")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    coach_query = st.text_area(
                        "What sales challenge can I help you with?",
                        placeholder="Examples:\n• How should I follow up with Rajesh at TechSolutions?\n• What's the best approach for pricing objections?\n• How can I improve my conversion rate from contacted to proposal?",
                        height=120
                    )
                
                with col2:
                    st.write("")  # Spacing
                    st.write("")
                    if st.button("Get Coach Advice", type="primary", use_container_width=True):
                        if coach_query:
                            with st.spinner("🧠 Analyzing your pipeline and crafting advice..."):
                                response = crm.sales_coach_chat(coach_query, daily_log, selected_rep)
                            st.markdown("### Coach's Advice")
                            st.markdown(f'<div class="coach-advice">{response}</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Please enter your question")
                    
                    if st.button("Get Today's Priorities", use_container_width=True):
                        with st.spinner("🔍 Analyzing your pipeline for today's focus areas..."):
                            analysis = crm.analyze_with_ai(daily_log, selected_rep)
                        st.markdown("### Today's Action Plan")
                        st.markdown(f'<div class="report-section">{analysis}</div>', unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Smart Message Generator")
                
                rep_leads = daily_log[daily_log['Sales Rep'] == selected_rep]
                if not rep_leads.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_lead = st.selectbox(
                            "Select Lead", 
                            rep_leads['Name'].unique(),
                            help="Choose the lead you want to message"
                        )
                    
                    with col2:
                        msg_type = st.selectbox(
                            "Message Type", 
                            ["Connection Message", "First Follow-up", "Proposal Follow-up", "Custom"],
                            help="Select the appropriate message context"
                        )
                    
                    if selected_lead:
                        lead_data = rep_leads[rep_leads['Name'] == selected_lead].iloc[0]
                        
                        # Lead info card
                        st.markdown("#### Lead Information")
                        info_cols = st.columns(3)
                        info_cols[0].write(f"**Company:** {lead_data['Company']}")
                        info_cols[1].write(f"**Title:** {lead_data['Title']}")
                        info_cols[2].write(f"**Status:** {lead_data.get('Status Stage', 'N/A')}")
                        
                        if st.button("Generate Message", type="primary"):
                            type_map = {
                                "Connection Message": "connection",
                                "First Follow-up": "follow_up_1", 
                                "Proposal Follow-up": "proposal_followup"
                            }
                            
                            with st.spinner("✍️ Crafting your message..."):
                                message = crm.generate_followup_message(lead_data, type_map.get(msg_type, "default"))
                            
                            st.markdown("#### Generated Message")
                            st.markdown(f'<div class="message-preview">{message}</div>', unsafe_allow_html=True)
                            
                            # Message actions
                            action_cols = st.columns(2)
                            with action_cols[0]:
                                st.download_button(
                                    "Download Message",
                                    message,
                                    file_name=f"message_{selected_lead.replace(' ', '_')}.txt",
                                    use_container_width=True
                                )
                            with action_cols[1]:
                                if st.button("Copy to Clipboard", use_container_width=True):
                                    st.success("Message copied to clipboard!")
                else:
                    st.info("No leads found for this sales rep")
            
            with tab3:
                st.subheader("Manager Report Generator")
                
                st.markdown("Create professional reports to share with your manager")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=7))
                with col2:
                    end_date = st.date_input("End Date", datetime.today())
                
                if st.button("Generate Manager Report", type="primary"):
                    with st.spinner("📊 Generating comprehensive performance report..."):
                        report = crm.generate_manager_report(
                            daily_log, 
                            selected_rep, 
                            start_date.strftime('%Y-%m-%d'), 
                            end_date.strftime('%Y-%m-%d')
                        )
                    
                    st.markdown("#### Generated Report")
                    st.markdown(f'<div class="report-section">{report}</div>', unsafe_allow_html=True)
                    
                    # WhatsApp Integration
                    st.markdown("---")
                    st.markdown("#### Send to Manager")
                    
                    whatsapp_url = send_whatsapp_message(MANAGER_WHATSAPP, report)
                    
                    st.markdown(f"""
                    **One-click WhatsApp Sharing**
                    
                    [📱 Click here to send report via WhatsApp]({whatsapp_url})
                    
                    *This will open WhatsApp with the report pre-filled and ready to send*
                    """)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h2>Welcome to Your AI Sales Assistant</h2>
            <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
                Upload your sales data to unlock AI-powered insights and automation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Smart Prioritization
            - AI identifies your top 3 daily priorities
            - Actionable insights based on your pipeline
            - Clear next steps for each lead
            """)
        
        with col2:
            st.markdown("""
            ### 💬 Perfect Messaging  
            - Context-aware message generation
            - Multiple message types
            - Professional templates
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Manager Reports
            - One-click report generation
            - WhatsApp integration
            - Professional formatting
            """)

if __name__ == "__main__":
    main()