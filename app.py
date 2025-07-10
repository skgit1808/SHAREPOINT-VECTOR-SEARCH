import streamlit as st

# Try importing the search function with error handling
try:
    from vector_search import vector_search
except ImportError as e:
    st.error(f"❌ Could not import search logic: {e}")
    st.stop()

st.set_page_config(page_title="SharePoint Vector Search", layout="centered")
st.title("📂 SharePoint Vector Search")
st.markdown("Enter a query to search your dummy SharePoint folder using vector similarity.")

# Input box for user query
query = st.text_input("Type your search query here:")

if query:
    st.write("🔍 Running vector search...")
    try:
        results = vector_search(query)
        if results:
            st.success(f"✅ Found {len(results)} relevant document(s):")
            for doc in results:
                st.markdown(f"**📄 {doc['file']}**")
                st.markdown(f"`{doc['path']}`")
                st.code(doc['preview'])
        else:
            st.warning("⚠️ No results found.")
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
